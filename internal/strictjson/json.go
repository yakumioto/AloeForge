// Package strictjson rejects duplicate keys before decoding bounded JSON metadata.
package strictjson

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strconv"
	"unicode/utf8"
)

const MaxDepth = 32

// Decode checks all nesting levels, rejects trailing data, and optionally
// rejects unknown struct fields. Diagnostics never include input text.
func Decode(data []byte, target any, rejectUnknown bool) error {
	if !utf8.Valid(data) {
		return errors.New("JSON is not UTF-8")
	}
	if !validUnicodeEscapes(data) {
		return errors.New("JSON contains an unpaired Unicode surrogate")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	if err := walk(d, 0); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("JSON has trailing data")
	}
	d = json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	if rejectUnknown {
		d.DisallowUnknownFields()
	}
	if err := d.Decode(target); err != nil {
		return errors.New("JSON fields or types are invalid")
	}
	return nil
}

// encoding/json replaces lone surrogates with U+FFFD; reject that ambiguity.
func validUnicodeEscapes(data []byte) bool {
	for i := 0; i < len(data); i++ {
		if data[i] != '\\' {
			continue
		}
		i++
		if i >= len(data) || data[i] != 'u' {
			continue
		}
		if i+4 >= len(data) {
			return false
		}
		value, err := strconv.ParseUint(string(data[i+1:i+5]), 16, 16)
		if err != nil {
			return false
		}
		i += 4
		if value >= 0xdc00 && value <= 0xdfff {
			return false
		}
		if value >= 0xd800 && value <= 0xdbff {
			if i+6 >= len(data) || data[i+1] != '\\' || data[i+2] != 'u' {
				return false
			}
			low, err := strconv.ParseUint(string(data[i+3:i+7]), 16, 16)
			if err != nil || low < 0xdc00 || low > 0xdfff {
				return false
			}
			i += 6
		}
	}
	return true
}

func walk(d *json.Decoder, depth int) error {
	if depth > MaxDepth {
		return errors.New("JSON nesting limit exceeded")
	}
	token, err := d.Token()
	if err != nil {
		return errors.New("invalid JSON")
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := make(map[string]bool)
		for d.More() {
			key, err := d.Token()
			name, ok := key.(string)
			if err != nil || !ok || seen[name] {
				return errors.New("invalid or duplicate JSON key")
			}
			seen[name] = true
			if err := walk(d, depth+1); err != nil {
				return err
			}
		}
	case '[':
		for d.More() {
			if err := walk(d, depth+1); err != nil {
				return err
			}
		}
	default:
		return errors.New("invalid JSON delimiter")
	}
	end, err := d.Token()
	if err != nil || (delim == '{' && end != json.Delim('}')) || (delim == '[' && end != json.Delim(']')) {
		return errors.New("invalid JSON container")
	}
	return nil
}
