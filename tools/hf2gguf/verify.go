package main

// verify.go proves the emitted GGUF equals the reference: identical metadata
// (keys, types, values; verified on raw wire bytes) and identical tensor set,
// shapes, ggml types and per-tensor payload bytes.

import (
	"bytes"
	"flag"
	"fmt"
	"io"

	gguf "github.com/cymertek/go-gguf"
)

type verifyResult struct {
	MetadataChecked int
	TensorsChecked  int
	ByteEqual       int
	MetaMismatches  []string
	TensorMismatch  []string
	OrderDiffers    bool
}

func (r *verifyResult) OK() bool {
	return len(r.MetaMismatches) == 0 && len(r.TensorMismatch) == 0 && !r.OrderDiffers
}

func compareGGUF(refPath, gotPath string) (*verifyResult, error) {
	res := &verifyResult{}

	// --- metadata on raw wire bytes ---
	_, refKVs, err := readRawKV(refPath)
	if err != nil {
		return nil, fmt.Errorf("read ref kv: %w", err)
	}
	_, gotKVs, err := readRawKV(gotPath)
	if err != nil {
		return nil, fmt.Errorf("read got kv: %w", err)
	}
	refMap := map[string]rawKV{}
	for _, kv := range refKVs {
		refMap[kv.Key] = kv
	}
	gotMap := map[string]rawKV{}
	for _, kv := range gotKVs {
		gotMap[kv.Key] = kv
	}
	for i, rk := range refKVs {
		gk, ok := gotMap[rk.Key]
		if !ok {
			res.MetaMismatches = append(res.MetaMismatches, fmt.Sprintf("missing key %q", rk.Key))
			continue
		}
		if i < len(gotKVs) && gotKVs[i].Key != rk.Key {
			res.OrderDiffers = true
		}
		switch {
		case gk.BType != rk.BType:
			res.MetaMismatches = append(res.MetaMismatches, fmt.Sprintf("%s: type %s != %s", rk.Key, gk.BType.name(), rk.BType.name()))
		case gk.rawKVHash() != rk.rawKVHash():
			res.MetaMismatches = append(res.MetaMismatches, fmt.Sprintf("%s: value bytes differ", rk.Key))
		}
		res.MetadataChecked++
	}
	for _, gk := range gotKVs {
		if _, ok := refMap[gk.Key]; !ok {
			res.MetaMismatches = append(res.MetaMismatches, fmt.Sprintf("extra key %q", gk.Key))
		}
	}

	// --- tensor metadata ---
	refT, err := loadTensorSet(refPath)
	if err != nil {
		return nil, err
	}
	gotT, err := loadTensorSet(gotPath)
	if err != nil {
		return nil, err
	}
	if len(refT.order) != len(gotT.order) {
		res.TensorMismatch = append(res.TensorMismatch, fmt.Sprintf("tensor count %d != %d", len(gotT.order), len(refT.order)))
	}
	for i, name := range refT.order {
		if i < len(gotT.order) && gotT.order[i] != name {
			res.OrderDiffers = true
		}
		ri := refT.byName[name]
		gi, ok := gotT.byName[name]
		if !ok {
			res.TensorMismatch = append(res.TensorMismatch, fmt.Sprintf("missing tensor %s", name))
			continue
		}
		if !shapesEqual(ri.Shape, gi.Shape) {
			res.TensorMismatch = append(res.TensorMismatch, fmt.Sprintf("%s: shape %v != %v", name, gi.Shape, ri.Shape))
			continue
		}
		if ri.GgmlType != gi.GgmlType {
			res.TensorMismatch = append(res.TensorMismatch, fmt.Sprintf("%s: type %s != %s", name, gi.GgmlType.GgmlName(), ri.GgmlType.GgmlName()))
			continue
		}
		res.TensorsChecked++
	}
	for _, name := range gotT.order {
		if _, ok := refT.byName[name]; !ok {
			res.TensorMismatch = append(res.TensorMismatch, fmt.Sprintf("extra tensor %s", name))
		}
	}

	// --- per-tensor payload bytes ---
	refG, err := gguf.Open(refPath)
	if err != nil {
		return nil, err
	}
	defer refG.Close()
	gotG, err := gguf.Open(gotPath)
	if err != nil {
		return nil, err
	}
	defer gotG.Close()

	refHandles := map[string]*gguf.Tensor{}
	for _, t := range mustTensors(refG) {
		refHandles[t.Info().Name] = t
	}
	gotHandles := map[string]*gguf.Tensor{}
	for _, t := range mustTensors(gotG) {
		gotHandles[t.Info().Name] = t
	}
	for _, name := range refT.order {
		rh, ok := refHandles[name]
		if !ok {
			continue
		}
		gh, ok := gotHandles[name]
		if !ok {
			continue
		}
		eq, diffOff, err := streamsEqual(rh.Reader(), gh.Reader())
		if err != nil {
			return nil, fmt.Errorf("compare %s: %w", name, err)
		}
		if !eq {
			res.TensorMismatch = append(res.TensorMismatch, fmt.Sprintf("%s: payload differs at byte %d", name, diffOff))
			continue
		}
		res.ByteEqual++
	}
	return res, nil
}

// tensorSet is an ordered view of a GGUF's tensor metadata.
type tensorSet struct {
	order  []string
	byName map[string]gguf.TensorInfo
}

func loadTensorSet(path string) (*tensorSet, error) {
	g, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	defer g.Close()
	ts, err := g.Tensors()
	if err != nil {
		return nil, err
	}
	out := &tensorSet{byName: map[string]gguf.TensorInfo{}}
	for _, t := range ts {
		info := t.Info()
		out.byName[info.Name] = info
		out.order = append(out.order, info.Name)
	}
	return out, nil
}

func mustTensors(g *gguf.GGUF) []*gguf.Tensor {
	ts, err := g.Tensors()
	if err != nil {
		panic(err)
	}
	return ts
}

func shapesEqual(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// streamsEqual compares two readers byte-by-byte, returning the offset of the
// first difference.
func streamsEqual(a, b io.Reader) (bool, int64, error) {
	const chunk = 1 << 20
	ba := make([]byte, chunk)
	bb := make([]byte, chunk)
	var off int64
	for {
		na, ea := io.ReadFull(a, ba)
		nb, eb := io.ReadFull(b, bb)
		if na != nb {
			return false, off + int64(min(na, nb)), nil
		}
		if !bytes.Equal(ba[:na], bb[:nb]) {
			for i := 0; i < na; i++ {
				if ba[i] != bb[i] {
					return false, off + int64(i), nil
				}
			}
		}
		off += int64(na)
		if ea == io.EOF || ea == io.ErrUnexpectedEOF {
			endB := eb == io.EOF || eb == io.ErrUnexpectedEOF
			return endB, off, nil
		}
	}
}

func cmdVerify(args []string) error {
	fs := flag.NewFlagSet("verify", flag.ExitOnError)
	ref := fs.String("ref", "", "reference GGUF")
	got := fs.String("got", "", "candidate GGUF")
	fs.Parse(args)
	if *ref == "" || *got == "" {
		return fmt.Errorf("verify: -ref and -got are required")
	}
	res, err := compareGGUF(*ref, *got)
	if err != nil {
		return err
	}
	fmt.Printf("metadata: %d checked, %d mismatches\n", res.MetadataChecked, len(res.MetaMismatches))
	fmt.Printf("tensors:  %d checked, %d byte-identical, %d mismatches\n", res.TensorsChecked, res.ByteEqual, len(res.TensorMismatch))
	if res.OrderDiffers {
		fmt.Println("WARNING: metadata or tensor order differs from reference")
	}
	for _, m := range res.MetaMismatches {
		fmt.Println("  META  ", m)
	}
	for _, m := range res.TensorMismatch {
		fmt.Println("  TENSOR", m)
	}
	if !res.OK() {
		return fmt.Errorf("verification FAILED")
	}
	fmt.Printf("verification PASSED: %d metadata keys and %d tensors byte-identical\n", res.MetadataChecked, res.ByteEqual)
	return nil
}
