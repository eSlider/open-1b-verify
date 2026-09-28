package main

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	gguf "github.com/cymertek/go-gguf"
)

func TestF32BitsToF16(t *testing.T) {
	cases := []struct {
		name string
		in   uint32
		want uint16
	}{
		{"zero", 0x00000000, 0x0000},
		{"negzero", 0x80000000, 0x8000},
		{"one", 0x3f800000, 0x3c00},
		{"neg2", 0xc0000000, 0xc000},
		{"half", 0x3f000000, 0x3800},
		{"max_f16", 0x477fe000, 0x7bff},
		{"overflow_to_inf", 0x477ff000, 0x7c00},
		{"inf", 0x7f800000, 0x7c00},
		{"neginf", 0xff800000, 0xfc00},
		{"qnan", 0x7fc00000, 0x7e00},
		{"snan", 0x7f800001, 0x7e00},
		{"min_normal", 0x38800000, 0x0400}, // 2^-14
		{"min_subnormal", 0x33800000, 0x0001}, // 2^-24
		{"tie_to_even_zero", 0x33000000, 0x0000}, // 2^-25 (half of min subnormal)
		{"tie_to_even_two", 0x33c00000, 0x0002},  // 3 * 2^-25
		{"one_tenth", 0x3dcccccd, 0x2e66},
	}
	for _, c := range cases {
		if got := f32bitsToF16(c.in); got != c.want {
			t.Errorf("%s: f32=%08x got %04x want %04x", c.name, c.in, got, c.want)
		}
	}
}

func TestF32ToF16Bytes(t *testing.T) {
	in := make([]byte, 8)
	putF32 := func(off int, f float32) {
		b := math.Float32bits(f)
		in[off] = byte(b)
		in[off+1] = byte(b >> 8)
		in[off+2] = byte(b >> 16)
		in[off+3] = byte(b >> 24)
	}
	putF32(0, 1.0)
	putF32(4, -2.0)
	out := f32ToF16Bytes(in)
	want := []byte{0x00, 0x3c, 0x00, 0xc0} // 1.0 -> 0x3c00 LE, -2.0 -> 0xc000 LE
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("byte %d: got %02x want %02x", i, out[i], want[i])
		}
	}
}

// TestRawKVRoundTrip creates a small GGUF with go-gguf, reads it with our raw
// walker, and re-writes the KVs through kvToValue to confirm arrays (including
// string arrays) survive byte-exactly.
func TestRawKVRoundTrip(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "a.gguf")
	dst := filepath.Join(dir, "b.gguf")

	w, err := gguf.OpenForWrite(src)
	if err != nil {
		t.Fatal(err)
	}
	strs := []byte{}
	for _, s := range []string{"alpha", "béta", ""} {
		strs = append(strs, leU64(uint64(len(s)))...)
		strs = append(strs, s...)
	}
	kvs := []gguf.KVEntry{
		{Key: "general.architecture", Value: gguf.Value{BType: gguf.BTypeString, Str: "open1b"}},
		{Key: "x.count", Value: gguf.Value{BType: gguf.BTypeUint32, Int: 7}},
		{Key: "x.f", Value: gguf.Value{BType: gguf.BTypeFloat32, Float: float64(float32(1e-5))}},
		{Key: "x.b", Value: gguf.Value{BType: gguf.BTypeBool, Int: 1}},
		{Key: "x.tags", Value: gguf.Value{BType: gguf.BTypeArray, ElemType: gguf.BTypeString, Int: 3, Raw: strs}},
	}
	for _, kv := range kvs {
		if err := w.SetKV(kv.Key, kv.Value); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := w.Close(); err != nil {
		t.Fatal(err)
	}

	_, got, err := readRawKV(src)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(kvs) {
		t.Fatalf("kv count %d want %d", len(got), len(kvs))
	}

	w2, err := gguf.OpenForWrite(dst)
	if err != nil {
		t.Fatal(err)
	}
	for _, kv := range got {
		v, err := kvToValue(kv)
		if err != nil {
			t.Fatal(err)
		}
		if err := w2.SetKV(kv.Key, v); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := w2.Close(); err != nil {
		t.Fatal(err)
	}

	_, round, err := readRawKV(dst)
	if err != nil {
		t.Fatal(err)
	}
	res, err := compareGGUF(src, dst)
	if err != nil {
		t.Fatal(err)
	}
	if !res.OK() {
		t.Fatalf("roundtrip not identical: meta=%v tensor=%v", res.MetaMismatches, res.TensorMismatch)
	}
	if len(round) != len(got) {
		t.Fatalf("rewritten kv count %d want %d", len(round), len(got))
	}
}

// TestVerifyRealFiles runs the full byte-level verification against the built
// artifacts when they are present in the workdir. Skipped when absent (fresh
// checkout) or in -short mode.
func TestVerifyRealFiles(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping multi-GB comparison in -short mode")
	}
	ref := "../models/open-1b-sft-f16.gguf"
	got := "../models/open-1b-sft-f16-go.gguf"
	if _, err := os.Stat(ref); err != nil {
		t.Skipf("reference GGUF not present: %v", err)
	}
	if _, err := os.Stat(got); err != nil {
		t.Skipf("candidate GGUF not present (run convert first): %v", err)
	}
	res, err := compareGGUF(ref, got)
	if err != nil {
		t.Fatal(err)
	}
	if !res.OK() {
		t.Fatalf("verification failed:\n metadata=%v\n tensors=%v", res.MetaMismatches, res.TensorMismatch)
	}
	if res.ByteEqual != res.TensorsChecked || res.TensorsChecked == 0 {
		t.Fatalf("expected all tensors byte-identical: %d/%d", res.ByteEqual, res.TensorsChecked)
	}
}
