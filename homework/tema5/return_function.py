def analyze_sentence(text):
    words = text.split()
    count = len(words)
    longest = max(words, key=len) if words else None
    shortest = min(words, key=len) if words else None
    avg_length = round(sum(len(word) for word in words) / len(words), 2) if words else 0
    contains_python = "da" if "python" in text.lower() else "nu"
    is_question = "da" if text.strip().endswith("?") else "nu"
    reversed_text = text[::-1]
    word_freq = {}
    for word in words:
        word_lower = word.lower()
        word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
    return {
        "word_count": count,
        "longest_word": longest,
        "shortest_word": shortest,
        "avg_word_length": avg_length,
        "contains_python": contains_python,
        "is_question": is_question,
        "reversed": reversed_text,
        "word_freq": word_freq
    }

def character_count(text, include_spaces=True):
    if include_spaces:
        return len(text)
    else:
        return len(text.replace(" ", ""))

def count_specific_word(text, target):
    return text.lower().split().count(target.lower())

def main():
    print("\nBuna! Acesta este un analizator de text complet.")
    
    while True:
        print(
            "\n--- Meniu ---\n"
            "1 - Analizeaza propozitia (numar cuvinte, cel mai lung cuvant, contine 'Python')\n"
            "2 - Cel mai scurt cuvant\n"
            "3 - Lungimea medie a cuvintelor\n"
            "4 - Numara un cuvant specific\n"
            "5 - Afiseaza propozitia cu majuscule\n"
            "6 - Afiseaza propozitia cu litere mici\n"
            "7 - Verifica daca propozitia este intrebare\n"
            "8 - Inversarea propozitiei\n"
            "9 - Numarul total de caractere\n"
            "10 - Frecventa fiecarui cuvant\n"
            "0 - Oprire program\n"
        )

        choice = input("Alegeti optiunea (0-10): ")

        if choice == "0":
            print("Programul s-a oprit. La revedere!")
            break
        elif choice in [str(i) for i in range(1, 11)]:
            sentence = input("Introduceti propozitia: ")
            analyzed = analyze_sentence(sentence)

            if choice == "1":
                print(f"\nNumar cuvinte: {analyzed['word_count']}")
                print(f"Cel mai lung cuvant: {analyzed['longest_word']}")
                print(f"Contine 'Python'? {analyzed['contains_python']}")
            elif choice == "2":
                print(f"Cel mai scurt cuvant: {analyzed['shortest_word']}")
            elif choice == "3":
                print(f"Lungimea medie a cuvintelor: {analyzed['avg_word_length']}")
            elif choice == "4":
                target = input("Introduceti cuvantul pe care doriti sa-l numarati: ")
                freq = count_specific_word(sentence, target)
                print(f"Cuvantul '{target}' apare de {freq} ori.")
            elif choice == "5":
                print(f"\nMajuscule: {sentence.upper()}")
            elif choice == "6":
                print(f"\nLitere mici: {sentence.lower()}")
            elif choice == "7":
                print(f"\nEste intrebare? {analyzed['is_question']}")
            elif choice == "8":
                print(f"\nPropozitia inversata: {analyzed['reversed']}")
            elif choice == "9":
                include_spaces = input("Sa includem spatiile? (da/nu): ").lower() == "da"
                char_count = character_count(sentence, include_spaces)
                print(f"\nNumarul total de caractere: {char_count}")
            elif choice == "10":
                print("\nFrecventa cuvintelor:")
                for word, freq in analyzed['word_freq'].items():
                    print(f"{word}: {freq}")
        else:
            print("\nOptiune invalida! Alegeti un numar intre 0 si 10.\n")

if __name__ == "__main__":
    main()