package main

// rawkv.go implements a minimal read-only GGUF metadata (KV) section walker.
//
// Why not go-gguf for this: the library's MetadataEntry.Value() deliberately
// returns an empty Value for BTypeArray (see readLazyValue in lazy.go), which is
// fine for scalar/string consumers but loses the tokenizer arrays
// (tokenizer.ggml.tokens / scores / token_type / merges). To reproduce the
// reference file's metadata byte-for-byte we need the raw array element bytes.
// go-gguf is still used for all tensor reading/writing and for scalar metadata;
// this walker only fills the array gap. It never writes GGUF.

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"os"
)

// ggufBType wire enum (mirrors gguf.BType).
type ggufBType uint32

const (
	btUint8   ggufBType = 0
	btInt8    ggufBType = 1
	btUint16  ggufBType = 2
	btInt16   ggufBType = 3
	btUint32  ggufBType = 4
	btInt32   ggufBType = 5
	btFloat32 ggufBType = 6
	btBool    ggufBType = 7
	btString  ggufBType = 8
	btArray   ggufBType = 9
	btUint64  ggufBType = 10
	btInt64   ggufBType = 11
	btFloat64 ggufBType = 12
)

func (t ggufBType) name() string {
	switch t {
	case btUint8:
		return "UINT8"
	case btInt8:
		return "INT8"
	case btUint16:
		return "UINT16"
	case btInt16:
		return "INT16"
	case btUint32:
		return "UINT32"
	case btInt32:
		return "INT32"
	case btFloat32:
		return "FLOAT32"
	case btBool:
		return "BOOL"
	case btString:
		return "STRING"
	case btArray:
		return "ARRAY"
	case btUint64:
		return "UINT64"
	case btInt64:
		return "INT64"
	case btFloat64:
		return "FLOAT64"
	default:
		return fmt.Sprintf("BTYPE(%d)", uint32(t))
	}
}

func (t ggufBType) size() int {
	switch t {
	case btUint8, btInt8, btBool:
		return 1
	case btUint16, btInt16:
		return 2
	case btUint32, btInt32, btFloat32:
		return 4
	case btUint64, btInt64, btFloat64:
		return 8
	default:
		return 0
	}
}

// rawKV is one metadata entry with its raw wire value preserved.
type rawKV struct {
	Key      string
	BType    ggufBType
	Int      int64     // integer/bool scalars
	Float    float64   // float scalars
	Str      string    // string scalars
	ElemType ggufBType // arrays: element type
	Count    uint64    // arrays: element count
	Raw      []byte    // arrays: concatenated element bytes (excludes elem_type+count)
}

// rawKVHeader holds the GGUF header fields.
type rawKVHeader struct {
	Version  uint32
	NTensors uint64
	NKV      uint64
}

// readRawKV opens path and returns the header plus every KV entry in file order.
func readRawKV(path string) (rawKVHeader, []rawKV, error) {
	f, err := os.Open(path)
	if err != nil {
		return rawKVHeader{}, nil, err
	}
	defer f.Close()

	var hdr rawKVHeader
	var b [24]byte
	if _, err := io.ReadFull(f, b[:]); err != nil {
		return hdr, nil, fmt.Errorf("read header: %w", err)
	}
	if string(b[0:4]) != "GGUF" {
		return hdr, nil, fmt.Errorf("bad magic %q", b[0:4])
	}
	hdr.Version = binary.LittleEndian.Uint32(b[4:8])
	hdr.NTensors = binary.LittleEndian.Uint64(b[8:16])
	hdr.NKV = binary.LittleEndian.Uint64(b[16:24])

	br := newBufReader(f)
	out := make([]rawKV, 0, hdr.NKV)
	for i := uint64(0); i < hdr.NKV; i++ {
		kv, err := readOneKV(br)
		if err != nil {
			return hdr, nil, fmt.Errorf("kv[%d]: %w", i, err)
		}
		out = append(out, kv)
	}
	return hdr, out, nil
}

func readOneKV(br *bufReader) (rawKV, error) {
	var kv rawKV
	key, err := br.str()
	if err != nil {
		return kv, fmt.Errorf("key: %w", err)
	}
	kv.Key = key
	bt, err := br.u32()
	if err != nil {
		return kv, fmt.Errorf("btype: %w", err)
	}
	kv.BType = ggufBType(bt)

	switch kv.BType {
	case btBool, btUint8, btInt8:
		v, err := br.u8()
		if err != nil {
			return kv, err
		}
		kv.Int = int64(v)
	case btUint16, btInt16:
		v, err := br.u16()
		if err != nil {
			return kv, err
		}
		if kv.BType == btInt16 {
			kv.Int = int64(int16(v))
		} else {
			kv.Int = int64(v)
		}
	case btUint32, btInt32:
		v, err := br.u32()
		if err != nil {
			return kv, err
		}
		if kv.BType == btInt32 {
			kv.Int = int64(int32(v))
		} else {
			kv.Int = int64(v)
		}
	case btFloat32:
		v, err := br.u32()
		if err != nil {
			return kv, err
		}
		kv.Float = float64(math.Float32frombits(v))
	case btUint64, btInt64:
		v, err := br.u64()
		if err != nil {
			return kv, err
		}
		kv.Int = int64(v)
	case btFloat64:
		v, err := br.u64()
		if err != nil {
			return kv, err
		}
		kv.Float = math.Float64frombits(v)
	case btString:
		s, err := br.str()
		if err != nil {
			return kv, err
		}
		kv.Str = s
	case btArray:
		et, err := br.u32()
		if err != nil {
			return kv, fmt.Errorf("array elem_type: %w", err)
		}
		kv.ElemType = ggufBType(et)
		cnt, err := br.u64()
		if err != nil {
			return kv, fmt.Errorf("array count: %w", err)
		}
		kv.Count = cnt
		raw, err := readArrayElems(br, kv.ElemType, cnt)
		if err != nil {
			return kv, fmt.Errorf("array elems: %w", err)
		}
		kv.Raw = raw
	default:
		return kv, fmt.Errorf("unsupported btype %d", kv.BType)
	}
	return kv, nil
}

// readArrayElems reads `count` elements of elemType, returning their concatenated
// wire bytes (exactly the bytes go-gguf's writer expects in Value.Raw).
func readArrayElems(br *bufReader, elemType ggufBType, count uint64) ([]byte, error) {
	if elemType == btArray || elemType == btString {
		// strings (and nested arrays, unsupported) are variable length
		if elemType != btString {
			return nil, fmt.Errorf("nested array element type unsupported")
		}
		var out []byte
		for i := uint64(0); i < count; i++ {
			slen, err := br.u64()
			if err != nil {
				return nil, err
			}
			out = append(out, leU64(slen)...)
			buf := make([]byte, slen)
			if _, err := io.ReadFull(br.r, buf); err != nil {
				return nil, err
			}
			out = append(out, buf...)
		}
		return out, nil
	}
	sz := elemType.size()
	if sz == 0 {
		return nil, fmt.Errorf("unknown array element type %d", elemType)
	}
	total := int(count) * sz
	// sanity cap: 256 MiB
	if uint64(total) > 256<<20 {
		return nil, fmt.Errorf("array too large: %d bytes", total)
	}
	buf := make([]byte, total)
	if _, err := io.ReadFull(br.r, buf); err != nil {
		return nil, err
	}
	return buf, nil
}

// rawKVHash is a stable fingerprint of a KV entry's full wire value.
func (kv rawKV) rawKVHash() string {
	h := crc32.NewIEEE()
	h.Write([]byte(kv.Key))
	var b [8]byte
	binary.LittleEndian.PutUint32(b[:4], uint32(kv.BType))
	h.Write(b[:4])
	switch kv.BType {
	case btBool, btUint8, btInt8, btUint16, btInt16, btUint32, btInt32, btUint64, btInt64:
		binary.LittleEndian.PutUint64(b[:], uint64(kv.Int))
		h.Write(b[:])
	case btFloat32:
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(float32(kv.Float)))
		h.Write(b[:4])
	case btFloat64:
		binary.LittleEndian.PutUint64(b[:], math.Float64bits(kv.Float))
		h.Write(b[:])
	case btString:
		h.Write([]byte(kv.Str))
	case btArray:
		binary.LittleEndian.PutUint32(b[:4], uint32(kv.ElemType))
		h.Write(b[:4])
		binary.LittleEndian.PutUint64(b[:], kv.Count)
		h.Write(b[:])
		h.Write(kv.Raw)
	}
	return fmt.Sprintf("%08x", h.Sum32())
}

// --- tiny buffered reader helpers ---

type bufReader struct {
	r io.Reader
}

func newBufReader(r io.Reader) *bufReader { return &bufReader{r: r} }

func (b *bufReader) u8() (uint8, error) {
	var x [1]byte
	if _, err := io.ReadFull(b.r, x[:]); err != nil {
		return 0, err
	}
	return x[0], nil
}

func (b *bufReader) u16() (uint16, error) {
	var x [2]byte
	if _, err := io.ReadFull(b.r, x[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint16(x[:]), nil
}

func (b *bufReader) u32() (uint32, error) {
	var x [4]byte
	if _, err := io.ReadFull(b.r, x[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(x[:]), nil
}

func (b *bufReader) u64() (uint64, error) {
	var x [8]byte
	if _, err := io.ReadFull(b.r, x[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(x[:]), nil
}

func (b *bufReader) str() (string, error) {
	n, err := b.u64()
	if err != nil {
		return "", err
	}
	if n > 1<<28 {
		return "", fmt.Errorf("string too long: %d", n)
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(b.r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func leU64(v uint64) []byte {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	return b[:]
}
