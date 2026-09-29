package main

// safetensors.go reads a minimal, read-only HuggingFace safetensors index.
//
// Format: [u64 LE header_len][header_len bytes of JSON][tensor data ...].
// The JSON header maps tensor name -> {dtype, shape, data_offsets:[start,end]}
// with data_offsets relative to the start of the data section.

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

type stTensor struct {
	Dtype string  `json:"dtype"`
	Shape []int64 `json:"shape"`
	Start int64   `json:"-"`
	End   int64   `json:"-"`
}

type stHeader struct {
	Dtype       string          `json:"dtype"`
	Shape       []int64         `json:"shape"`
	DataOffsets [2]int64        `json:"data_offsets"`
	Extra       json.RawMessage `json:"-"`
}

type safetensors struct {
	path      string
	f         *os.File
	dataStart int64
	Meta      map[string]string
	Order     []string
	Tensors   map[string]stTensor
}

func openSafetensors(path string) (*safetensors, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var lenBuf [8]byte
	if _, err := io.ReadFull(f, lenBuf[:]); err != nil {
		f.Close()
		return nil, fmt.Errorf("read header length: %w", err)
	}
	hlen := binary.LittleEndian.Uint64(lenBuf[:])
	if hlen == 0 || hlen > 1<<30 {
		f.Close()
		return nil, fmt.Errorf("implausible safetensors header length %d", hlen)
	}
	raw := make([]byte, hlen)
	if _, err := io.ReadFull(f, raw); err != nil {
		f.Close()
		return nil, fmt.Errorf("read header json: %w", err)
	}

	// Top-level: either "__metadata__" (object of strings) or a tensor descriptor.
	var top map[string]json.RawMessage
	if err := json.Unmarshal(raw, &top); err != nil {
		f.Close()
		return nil, fmt.Errorf("parse header json: %w", err)
	}

	st := &safetensors{
		path:      path,
		f:         f,
		dataStart: int64(8 + hlen),
		Meta:      map[string]string{},
		Tensors:   map[string]stTensor{},
	}
	if metaRaw, ok := top["__metadata__"]; ok {
		var m map[string]string
		if err := json.Unmarshal(metaRaw, &m); err == nil {
			st.Meta = m
		}
		delete(top, "__metadata__")
	}
	for name, rawDesc := range top {
		var h stHeader
		if err := json.Unmarshal(rawDesc, &h); err != nil {
			f.Close()
			return nil, fmt.Errorf("tensor %q: %w", name, err)
		}
		st.Tensors[name] = stTensor{
			Dtype: h.Dtype,
			Shape: h.Shape,
			Start: h.DataOffsets[0],
			End:   h.DataOffsets[1],
		}
		st.Order = append(st.Order, name)
	}
	return st, nil
}

func (s *safetensors) Close() error {
	if s.f != nil {
		return s.f.Close()
	}
	return nil
}

// readTensor returns the raw little-endian bytes of a tensor.
func (s *safetensors) readTensor(name string) ([]byte, stTensor, error) {
	t, ok := s.Tensors[name]
	if !ok {
		return nil, stTensor{}, fmt.Errorf("no tensor %q", name)
	}
	n := t.End - t.Start
	if n < 0 {
		return nil, t, fmt.Errorf("tensor %q: negative size", name)
	}
	buf := make([]byte, n)
	if _, err := s.f.ReadAt(buf, s.dataStart+t.Start); err != nil {
		return nil, t, fmt.Errorf("read %q: %w", name, err)
	}
	return buf, t, nil
}

// numElems returns the product of shape dimensions.
func numElems(shape []int64) int64 {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	return n
}

// stDtypeBytes returns bytes per element for the supported safetensors dtypes.
func stDtypeBytes(dtype string) (int, error) {
	switch dtype {
	case "F32":
		return 4, nil
	case "F16":
		return 2, nil
	case "BF16":
		return 2, nil
	case "I64":
		return 8, nil
	case "I32":
		return 4, nil
	case "I8":
		return 1, nil
	case "U8":
		return 1, nil
	case "F64":
		return 8, nil
	default:
		return 0, fmt.Errorf("unsupported safetensors dtype %q", dtype)
	}
}
