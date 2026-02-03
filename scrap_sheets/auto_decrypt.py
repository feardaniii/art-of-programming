import string

PRINTABLE = set(string.printable)


def parse_input(raw):
    raw = raw.strip().replace(",", " ").replace("[", "").replace("]", "")
    return [int(x) for x in raw.split()]


def is_printable(text, threshold=0.9):
    if not text:
        return False
    printable_count = sum(c in PRINTABLE for c in text)
    return printable_count / len(text) >= threshold


# ---------- DECODERS ----------

def ascii_direct(arr):
    try:
        text = ''.join(chr(n) for n in arr)
        if is_printable(text):
            return text
    except:
        pass


def ascii_caesar(arr):
    results = []
    for shift in range(-20, 21):
        try:
            text = ''.join(chr(n + shift) for n in arr)
            if is_printable(text):
                results.append(f"shift {shift:+}: {text}")
        except:
            pass
    return results


def alpha_direct(arr):
    text = ""
    for n in arr:
        if n == 0:
            text += " "
        elif 1 <= n <= 26:
            text += chr(ord("a") + n - 1)
        else:
            return None
    return text


def alpha_caesar(arr):
    results = []
    for shift in range(26):
        text = ""
        valid = True
        for n in arr:
            if n == 0:
                text += " "
            elif 1 <= n <= 26:
                v = (n - 1 + shift) % 26
                text += chr(ord("a") + v)
            else:
                valid = False
                break
        if valid and is_printable(text):
            results.append(f"shift {shift}: {text}")
    return results


def xor_single_byte(arr):
    results = []
    for key in range(256):
        try:
            text = ''.join(chr(n ^ key) for n in arr)
            if is_printable(text):
                results.append(f"key {key}: {text}")
        except:
            pass
    return results


# ---------- MAIN ----------

def main():
    print("🔐 Auto Decryptor")
    print("Paste encrypted numeric array:")
    raw = input("> ")

    arr = parse_input(raw)
    print(f"\nLoaded {len(arr)} numbers")
    print("-" * 40)

    print("\n[+] ASCII direct:")
    out = ascii_direct(arr)
    if out:
        print(out)

    print("\n[+] ASCII Caesar shifts:")
    for r in ascii_caesar(arr):
        print(r)

    print("\n[+] Alphabet (A=1..Z=26):")
    out = alpha_direct(arr)
    if out:
        print(out)

    print("\n[+] Alphabet Caesar shifts:")
    for r in alpha_caesar(arr):
        print(r)

    print("\n[+] XOR single-byte brute force:")
    for r in xor_single_byte(arr)[:20]:  # limit spam
        print(r)

    print("\nDone.")


if __name__ == "__main__":
    main()
