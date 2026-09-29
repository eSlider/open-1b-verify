package main

// convert.go converts the HF safetensors into models/open-1b-sft-f16-go.gguf,
// reproducing the reference GGUF's metadata (verbatim) and tensor set/order.
//
// Tensor mapping (HF FQN -> GGUF name), shapes are the reference's (ggml ne0 is
// the innermost/HF input dim, so the HF bytes are used unpermuted):
//
//	embedding.tok_embeddings.weight  -> token_embd.weight
//	emb_norm.weight                  -> token_embd_norm.weight
//	embedding.output.weight          -> output.weight
//	norm_out.weight                  -> output_norm.weight
//	blocks.{i}.attn.wq.weight        -> blk.{i}.attn_q.weight
//	blocks.{i}.attn.wk.weight        -> blk.{i}.attn_k.weight
//	blocks.{i}.attn.wv.weight        -> blk.{i}.attn_v.weight
//	blocks.{i}.attn.wo.weight        -> blk.{i}.attn_output.weight
//	blocks.{i}.ffn.w_gate_up.weight  -> blk.{i}.ffn_gate.weight (first half)
//	                                 -> blk.{i}.ffn_up.weight   (second half)
//	blocks.{i}.ffn.w_down.weight     -> blk.{i}.ffn_down.weight
//	blocks.{i}.norm1.weight          -> blk.{i}.attn_norm.weight
//	blocks.{i}.norm2.weight          -> blk.{i}.ffn_norm.weight
//
// The quantized per-channel weight_scale tensors in the checkpoint are dropped,
// matching the reference.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"regexp"

	gguf "github.com/cymertek/go-gguf"
)

// hfRef resolves a GGUF tensor name to a HF safetensors tensor plus an optional
// element range [start,end) (for the fused gate+up split). end=-1 means "all".
type hfRef struct {
	Name  string
	Start int64
	End   int64
}

func resolveHF(refName string) (hfRef, error) {
	switch refName {
	case "token_embd.weight":
		return hfRef{"embedding.tok_embeddings.weight", 0, -1}, nil
	case "token_embd_norm.weight":
		return hfRef{"emb_norm.weight", 0, -1}, nil
	case "output.weight":
		return hfRef{"embedding.output.weight", 0, -1}, nil
	case "output_norm.weight":
		return hfRef{"norm_out.weight", 0, -1}, nil
	}
	m := blkRe.FindStringSubmatch(refName)
	if m == nil {
		return hfRef{}, fmt.Errorf("unmapped tensor %q", refName)
	}
	i := m[1]
	leaf := m[2]
	switch leaf {
	case "attn_q":
		return hfRef{"blocks." + i + ".attn.wq.weight", 0, -1}, nil
	case "attn_k":
		return hfRef{"blocks." + i + ".attn.wk.weight", 0, -1}, nil
	case "attn_v":
		return hfRef{"blocks." + i + ".attn.wv.weight", 0, -1}, nil
	case "attn_output":
		return hfRef{"blocks." + i + ".attn.wo.weight", 0, -1}, nil
	case "ffn_gate":
		// fused w_gate_up is [2*ffn, hidden]; chunk(2) => gate = first half
		return hfRef{"blocks." + i + ".ffn.w_gate_up.weight", 0, 0}, nil // end filled later
	case "ffn_up":
		return hfRef{"blocks." + i + ".ffn.w_gate_up.weight", 0, 0}, nil
	case "ffn_down":
		return hfRef{"blocks." + i + ".ffn.w_down.weight", 0, -1}, nil
	case "attn_norm":
		return hfRef{"blocks." + i + ".norm1.weight", 0, -1}, nil
	case "ffn_norm":
		return hfRef{"blocks." + i + ".norm2.weight", 0, -1}, nil
	}
	return hfRef{}, fmt.Errorf("unmapped block tensor %q", refName)
}

var blkRe = regexp.MustCompile(`^blk\.(\d+)\.(.+)\.weight$`)

func cmdConvert(args []string) error {
	fs := flag.NewFlagSet("convert", flag.ExitOnError)
	refGGUF := fs.String("ref", "", "reference GGUF (metadata/tensor contract)")
	stPath := fs.String("safetensors", "", "HF model.safetensors")
	hfConfig := fs.String("config", "", "HF config.json (validated against ref)")
	hfTokConfig := fs.String("tokenizer-config", "", "HF tokenizer_config.json (validated against ref)")
	hfTokenizer := fs.String("tokenizer", "", "HF tokenizer.json (validated against ref)")
	outPath := fs.String("o", "", "output GGUF path")
	fs.Parse(args)
	if *refGGUF == "" || *stPath == "" || *outPath == "" {
		return fmt.Errorf("convert: -ref, -safetensors and -o are required")
	}

	hdr, kvs, err := readRawKV(*refGGUF)
	if err != nil {
		return fmt.Errorf("read reference metadata: %w", err)
	}
	ref, err := gguf.Open(*refGGUF)
	if err != nil {
		return fmt.Errorf("open reference: %w", err)
	}
	refTensors, err := ref.Tensors()
	if err != nil {
		return fmt.Errorf("reference tensors: %w", err)
	}

	st, err := openSafetensors(*stPath)
	if err != nil {
		return fmt.Errorf("open safetensors: %w", err)
	}
	defer st.Close()

	// Validate HF config against the reference contract (no values are taken from
	// it (the reference is authoritative), but mismatches are fatal).
	if *hfConfig != "" {
		if err := validateConfig(*hfConfig, kvs); err != nil {
			return fmt.Errorf("config validation: %w", err)
		}
	}
	if *hfTokConfig != "" {
		if err := validateTokenizerConfig(*hfTokConfig, kvs); err != nil {
			return fmt.Errorf("tokenizer config validation: %w", err)
		}
	}
	if *hfTokenizer != "" {
		if err := validateTokenizerJSON(*hfTokenizer, kvs); err != nil {
			return fmt.Errorf("tokenizer.json validation: %w", err)
		}
	}

	w, err := gguf.OpenForWrite(*outPath)
	if err != nil {
		return fmt.Errorf("create output: %w", err)
	}
	// Metadata first: copy the reference KV section verbatim.
	for _, kv := range kvs {
		v, err := kvToValue(kv)
		if err != nil {
			w.Close()
			return fmt.Errorf("metadata %q: %w", kv.Key, err)
		}
		if err := w.SetKV(kv.Key, v); err != nil {
			w.Close()
			return fmt.Errorf("set %q: %w", kv.Key, err)
		}
	}

	// Tensors in reference order so the data layout matches.
	byName := map[string]gguf.TensorInfo{}
	for _, t := range refTensors {
		byName[t.Info().Name] = t.Info()
	}
	for _, t := range refTensors {
		info := t.Info()
		r, err := resolveHF(info.Name)
		if err != nil {
			w.Close()
			return err
		}
		hfT, ok := st.Tensors[r.Name]
		if !ok {
			w.Close()
			return fmt.Errorf("tensor %s: HF source %q missing", info.Name, r.Name)
		}
		elemBytes, err := stDtypeBytes(hfT.Dtype)
		if err != nil {
			w.Close()
			return fmt.Errorf("tensor %s: %w", info.Name, err)
		}
		total := numElems(hfT.Shape)
		if hfT.Dtype != "F32" && hfT.Dtype != "F16" {
			w.Close()
			return fmt.Errorf("tensor %s: source dtype %s unsupported", info.Name, hfT.Dtype)
		}
		// Resolve the fused gate/up split now that we know the element count.
		start, end := r.Start, r.End
		if end == -1 {
			end = total
		}
		if isFusedSplit(info.Name) {
			half := total / 2
			switch {
			case hasSuffix(info.Name, "ffn_gate.weight"):
				start, end = 0, half
			case hasSuffix(info.Name, "ffn_up.weight"):
				start, end = half, total
			}
		}
		if end > total || start < 0 || start >= end {
			w.Close()
			return fmt.Errorf("tensor %s: bad element range [%d,%d) of %d", info.Name, start, end, total)
		}
		wantBytes := uint64(end-start) * uint64(info.GgmlType.BlockBytes()) / uint64(info.GgmlType.ElementsPerBlock())
		if info.NBytes != wantBytes {
			w.Close()
			return fmt.Errorf("tensor %s: ref nbytes %d != expected %d", info.Name, info.NBytes, wantBytes)
		}

		// Shape check: reference shape must be the HF shape reversed (ggml ne0
		// innermost) and consistent with the slice.
		if !shapeMatches(info.Shape, hfT.Shape, start, end, total) {
			w.Close()
			return fmt.Errorf("tensor %s: ref shape %v vs HF %v (range %d:%d)", info.Name, info.Shape, hfT.Shape, start, end)
		}

		dataOff := st.dataStart + hfT.Start + start*int64(elemBytes)
		rem := (end - start) * int64(elemBytes)

		idx := w.AddTensor(info.Name, info.Shape, info.GgmlType)
		var rd io.Reader
		switch info.GgmlType {
		case gguf.GgmlF16:
			rd = &f16StreamReader{f: st.f, pos: dataOff, rem: rem}
		case gguf.GgmlF32:
			rd = &f32StreamReader{f: st.f, pos: dataOff, rem: rem}
		default:
			w.Close()
			return fmt.Errorf("tensor %s: unsupported target type %s", info.Name, info.GgmlType.GgmlName())
		}
		if err := w.WriteTensorData(idx, rd); err != nil {
			w.Close()
			return fmt.Errorf("tensor %s: %w", info.Name, err)
		}
	}

	n, err := w.Close()
	if err != nil {
		return fmt.Errorf("finalize: %w", err)
	}
	fmt.Printf("wrote %s: %d bytes, %d metadata keys, %d tensors (ref nKV=%d nTensors=%d)\n",
		*outPath, n, len(kvs), len(refTensors), hdr.NKV, hdr.NTensors)
	return nil
}

func isFusedSplit(name string) bool {
	return hasSuffix(name, "ffn_gate.weight") || hasSuffix(name, "ffn_up.weight")
}

func hasSuffix(s, suffix string) bool {
	return len(s) >= len(suffix) && s[len(s)-len(suffix):] == suffix
}

// shapeMatches verifies ref ggml shape against the HF shape for a slice.
// Whole tensors: ref == reverse(hf). A gate/up half: ref == [hf[1], hf[0]/2].
func shapeMatches(ref []uint64, hf []int64, start, end, total int64) bool {
	if len(ref) != len(hf) {
		return false
	}
	if end-start != total {
		// fused split: both ref and hf are 2-D; ref[0]==hf[1],
		// ref[1]==hf[0]/2 and the split is on the outer (row) dim.
		if len(ref) != 2 {
			return false
		}
		return ref[0] == uint64(hf[1]) && ref[1] == uint64(hf[0]/2) && hf[0]%2 == 0
	}
	for i := range ref {
		if ref[i] != uint64(hf[len(hf)-1-i]) {
			return false
		}
	}
	return true
}

// ---------------------------------------------------------------------------
// streaming readers that convert F32 -> F16 on the fly (RNE)
// ---------------------------------------------------------------------------

const streamChunk = 1 << 20 // 1 MiB of source bytes per read

type f16StreamReader struct {
	f       *os.File
	pos     int64
	rem     int64
	pending []byte
}

func (r *f16StreamReader) Read(p []byte) (int, error) {
	for len(r.pending) == 0 {
		if r.rem <= 0 {
			return 0, io.EOF
		}
		n := r.rem
		if n > streamChunk {
			n = streamChunk
		}
		n -= n % 4 // whole f32 elements
		buf := make([]byte, n)
		if _, err := r.f.ReadAt(buf, r.pos); err != nil {
			return 0, err
		}
		r.pos += n
		r.rem -= n
		r.pending = f32ToF16Bytes(buf)
	}
	n := copy(p, r.pending)
	r.pending = r.pending[n:]
	return n, nil
}

type f32StreamReader struct {
	f       *os.File
	pos     int64
	rem     int64
	pending []byte
}

func (r *f32StreamReader) Read(p []byte) (int, error) {
	for len(r.pending) == 0 {
		if r.rem <= 0 {
			return 0, io.EOF
		}
		n := r.rem
		if n > streamChunk {
			n = streamChunk
		}
		buf := make([]byte, n)
		if _, err := r.f.ReadAt(buf, r.pos); err != nil {
			return 0, err
		}
		r.pos += n
		r.rem -= n
		r.pending = buf
	}
	n := copy(p, r.pending)
	r.pending = r.pending[n:]
	return n, nil
}

// ---------------------------------------------------------------------------
// FP32 -> FP16 (IEEE-754 round-to-nearest-even), matching numpy/torch casts.
// ---------------------------------------------------------------------------

func f32ToF16Bytes(src []byte) []byte {
	out := make([]byte, len(src)/4*2)
	for i := 0; i+4 <= len(src); i += 4 {
		bits := uint32(src[i]) | uint32(src[i+1])<<8 | uint32(src[i+2])<<16 | uint32(src[i+3])<<24
		half := f32bitsToF16(bits)
		j := i / 4 * 2
		out[j] = byte(half)
		out[j+1] = byte(half >> 8)
	}
	return out
}

// f32bitsToF16 converts the raw bits of an IEEE-754 binary32 to binary16 using
// round-to-nearest, ties-to-even.
func f32bitsToF16(b uint32) uint16 {
	sign := uint16((b >> 16) & 0x8000)
	exp := int32((b >> 23) & 0xff)
	mant := b & 0x7fffff

	if exp == 0xff { // Inf or NaN
		if mant != 0 {
			return sign | 0x7e00 // quiet NaN
		}
		return sign | 0x7c00 // Inf
	}

	e := exp - 127 // unbiased

	if e > 15 { // overflow -> Inf
		return sign | 0x7c00
	}
	if e >= -14 { // normal
		hm := uint16(e+15)<<10 | uint16(mant>>13)
		rem := mant & 0x1fff
		if rem > 0x1000 || (rem == 0x1000 && hm&1 == 1) {
			hm++
		}
		return sign | hm
	}
	if e < -25 { // too small -> signed zero
		return sign
	}
	// subnormal: implicit leading 1, shift into place
	m := mant | 0x800000
	shift := uint32(-e - 14) // 1..11
	hm := uint16(m >> (13 + shift))
	rem := m & ((1 << (13 + shift)) - 1)
	halfBit := uint32(1) << (12 + shift)
	if rem > halfBit || (rem == halfBit && hm&1 == 1) {
		hm++
	}
	return sign | hm
}

// ---------------------------------------------------------------------------
// HF config / tokenizer config validation (reference is authoritative)
// ---------------------------------------------------------------------------
