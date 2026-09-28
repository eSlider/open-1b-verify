package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"

	gguf "github.com/cymertek/go-gguf"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: hf2gguf <dump|convert|verify> [flags]")
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "dump":
		err = cmdDump(os.Args[2:])
	case "convert":
		err = cmdConvert(os.Args[2:])
	case "verify":
		err = cmdVerify(os.Args[2:])
	default:
		err = fmt.Errorf("unknown subcommand %q", os.Args[1])
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// ---------------------------------------------------------------------------
// Manifest types — the reproduction contract.
// ---------------------------------------------------------------------------

type manifest struct {
	Source   string      `json:"source"`
	Version  uint32      `json:"version"`
	NTensors uint64      `json:"n_tensors"`
	NKV      uint64      `json:"n_kv"`
	Metadata []metaKV    `json:"metadata"`
	Tensors  []tensorRow `json:"tensors"`
}

type metaKV struct {
	Key       string    `json:"key"`
	Type      string    `json:"type"`
	Value     any       `json:"value,omitempty"`
	ElemType  string    `json:"elem_type,omitempty"`
	Count     uint64    `json:"count,omitempty"`
	RawBytes  int       `json:"raw_bytes,omitempty"`
	RawSHA256 string    `json:"raw_sha256,omitempty"`
	Preview   []string  `json:"preview,omitempty"`
	Numbers   []float64 `json:"numbers,omitempty"`
}

type tensorRow struct {
	Name   string   `json:"name"`
	Shape  []uint64 `json:"shape"`
	Type   string   `json:"type"`
	TypeID uint32   `json:"type_id"`
	NBytes uint64   `json:"nbytes"`
}

func buildManifest(ggufPath string) (*manifest, error) {
	hdr, kvs, err := readRawKV(ggufPath)
	if err != nil {
		return nil, err
	}
	m := &manifest{
		Source:   ggufPath,
		Version:  hdr.Version,
		NTensors: hdr.NTensors,
		NKV:      hdr.NKV,
	}
	for _, kv := range kvs {
		m.Metadata = append(m.Metadata, kvToManifest(kv))
	}

	g, err := gguf.Open(ggufPath)
	if err != nil {
		return nil, fmt.Errorf("go-gguf open: %w", err)
	}
	defer g.Close()
	tensors, err := g.Tensors()
	if err != nil {
		return nil, fmt.Errorf("go-gguf tensors: %w", err)
	}
	for _, t := range tensors {
		info := t.Info()
		m.Tensors = append(m.Tensors, tensorRow{
			Name:   info.Name,
			Shape:  info.Shape,
			Type:   info.GgmlType.GgmlName(),
			TypeID: uint32(info.GgmlType),
			NBytes: info.NBytes,
		})
	}
	if uint64(len(m.Tensors)) != hdr.NTensors {
		return nil, fmt.Errorf("tensor count mismatch: header %d, walked %d", hdr.NTensors, len(m.Tensors))
	}
	if uint64(len(m.Metadata)) != hdr.NKV {
		return nil, fmt.Errorf("kv count mismatch: header %d, walked %d", hdr.NKV, len(m.Metadata))
	}
	return m, nil
}

func kvToManifest(kv rawKV) metaKV {
	out := metaKV{Key: kv.Key, Type: kv.BType.name()}
	switch kv.BType {
	case btBool:
		out.Value = kv.Int != 0
	case btUint8, btInt8, btUint16, btInt16, btUint32, btInt32, btUint64, btInt64:
		out.Value = kv.Int
	case btFloat32:
		out.Value = float32(kv.Float)
	case btFloat64:
		out.Value = kv.Float
	case btString:
		out.Value = kv.Str
	case btArray:
		out.ElemType = kv.ElemType.name()
		out.Count = kv.Count
		out.RawBytes = len(kv.Raw)
		sum := sha256.Sum256(kv.Raw)
		out.RawSHA256 = fmt.Sprintf("%x", sum)
		decodePreview(kv, &out)
	}
	return out
}

// decodePreview fills Preview/Numbers for small arrays so the manifest stays readable.
func decodePreview(kv rawKV, out *metaKV) {
	if kv.Count > 64 {
		// give a tiny human preview even for huge token arrays
		if kv.ElemType == btString {
			out.Preview = decodeStringArray(kv.Raw, 3)
		}
		return
	}
	switch kv.ElemType {
	case btString:
		out.Preview = decodeStringArray(kv.Raw, kv.Count)
	case btFloat32:
		n := int(kv.Count)
		for i := 0; i < n; i++ {
			bits := binary.LittleEndian.Uint32(kv.Raw[i*4:])
			out.Numbers = append(out.Numbers, float64(math.Float32frombits(bits)))
		}
	case btInt32:
		n := int(kv.Count)
		for i := 0; i < n; i++ {
			out.Numbers = append(out.Numbers, float64(int32(binary.LittleEndian.Uint32(kv.Raw[i*4:]))))
		}
	case btUint32:
		n := int(kv.Count)
		for i := 0; i < n; i++ {
			out.Numbers = append(out.Numbers, float64(binary.LittleEndian.Uint32(kv.Raw[i*4:])))
		}
	}
}

func decodeStringArray(raw []byte, limit uint64) []string {
	var out []string
	off := 0
	for i := uint64(0); i < limit; i++ {
		if off+8 > len(raw) {
			break
		}
		n := binary.LittleEndian.Uint64(raw[off:])
		off += 8
		if off+int(n) > len(raw) {
			break
		}
		out = append(out, string(raw[off:off+int(n)]))
		off += int(n)
	}
	return out
}

// ---------------------------------------------------------------------------
// dump
// ---------------------------------------------------------------------------

func cmdDump(args []string) error {
	fs := flag.NewFlagSet("dump", flag.ExitOnError)
	ggufPath := fs.String("gguf", "", "reference GGUF path")
	outPath := fs.String("o", "", "output JSON manifest path (default stdout)")
	fs.Parse(args)
	if *ggufPath == "" {
		return fmt.Errorf("dump: -gguf is required")
	}
	m, err := buildManifest(*ggufPath)
	if err != nil {
		return err
	}
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if *outPath == "" {
		_, err = os.Stdout.Write(b)
		return err
	}
	return os.WriteFile(*outPath, b, 0o644)
}

// ---------------------------------------------------------------------------
// shared helpers for convert/verify
// ---------------------------------------------------------------------------

// btypeFromName maps the manifest type name back to the raw enum.
func btypeFromName(s string) (ggufBType, error) {
	for _, t := range []ggufBType{btUint8, btInt8, btUint16, btInt16, btUint32,
		btInt32, btFloat32, btBool, btString, btArray, btUint64, btInt64, btFloat64} {
		if t.name() == s {
			return t, nil
		}
	}
	return 0, fmt.Errorf("unknown btype %q", s)
}

// kvToValue converts a rawKV into a go-gguf write Value, preserving the wire
// representation exactly (arrays keep ElemType + count + raw element bytes).
func kvToValue(kv rawKV) (gguf.Value, error) {
	switch kv.BType {
	case btBool:
		return gguf.Value{BType: gguf.BTypeBool, Int: kv.Int}, nil
	case btUint8:
		return gguf.Value{BType: gguf.BTypeUint8, Int: kv.Int}, nil
	case btInt8:
		return gguf.Value{BType: gguf.BTypeInt8, Int: kv.Int}, nil
	case btUint16:
		return gguf.Value{BType: gguf.BTypeUint16, Int: kv.Int}, nil
	case btInt16:
		return gguf.Value{BType: gguf.BTypeInt16, Int: kv.Int}, nil
	case btUint32:
		return gguf.Value{BType: gguf.BTypeUint32, Int: kv.Int}, nil
	case btInt32:
		return gguf.Value{BType: gguf.BTypeInt32, Int: kv.Int}, nil
	case btUint64:
		return gguf.Value{BType: gguf.BTypeUint64, Int: kv.Int}, nil
	case btInt64:
		return gguf.Value{BType: gguf.BTypeInt64, Int: kv.Int}, nil
	case btFloat32:
		return gguf.Value{BType: gguf.BTypeFloat32, Float: kv.Float}, nil
	case btFloat64:
		return gguf.Value{BType: gguf.BTypeFloat64, Float: kv.Float}, nil
	case btString:
		return gguf.Value{BType: gguf.BTypeString, Str: kv.Str}, nil
	case btArray:
		return gguf.Value{
			BType:    gguf.BTypeArray,
			ElemType: btypeToGguf(kv.ElemType),
			Int:      int64(kv.Count),
			Raw:      kv.Raw,
		}, nil
	default:
		return gguf.Value{}, fmt.Errorf("unsupported btype %d for key %q", kv.BType, kv.Key)
	}
}

func btypeToGguf(t ggufBType) gguf.BType {
	return gguf.BType(t)
}

func sha256Hex(b []byte) string {
	s := sha256.Sum256(b)
	return fmt.Sprintf("%x", s)
}

func shortHash(b []byte) string {
	s := sha256Hex(b)
	if len(s) > 12 {
		return s[:12]
	}
	return s
}
