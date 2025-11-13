# I. Adapteaza-ti un exercitiu la alegere din temele trecute, implementand functii.
# Foloseste-ti imaginatia, am putea pune fiecare linie intr-o functie separata, dar Arta este sa distingem acele parti
# cheie compuse din secvente de actiuni care compun o Actiune mai mare - Re-utilizabila, Organizata, Modulara
# Ex: Functie pentru meniu, functie pentru adaugare(), functie pentru filtrare, cautare, etc.

playlist = [
    {"title": "Highest In The Room", "artist": "Travis Scott", "genre": "Electronica", "duration": 177, "rating": 7},
    {"title": "Godzilla", "artist": "Eminem", "genre": "Hip-Hop", "duration": 267, "rating": 8},
    {"title": "Love Story", "artist": "Taylor Swift", "genre": "Country", "duration": 237, "rating": 9},
    {"title": "Let Me Down Slowly", "artist": "Alec Benjamin", "genre": "Alternative", "duration": 178, "rating": 8},
    {"title": "Fly Me to the Moon", "artist": "Frank Sinatra", "genre": "Jazz", "duration":148 , "rating": 8},
    {"title": "Symphony No. 5", "artist": "Beethoven", "genre": "Classical", "duration": 2880, "rating": 10}
]

def show_songs(playlist):
    for song in playlist:
        print(f"{song['title']} - {song['artist']} ({song['genre']}) "
              f"{song['duration']}s, rating: {song['rating']}")
        
def filter_by_genre(playlist, genre):
    found_genre = False
   
    filtered_songs = []

    for song in playlist:
        if song["genre"].lower() == genre.lower():
            filtered_songs.append(song)
            found_genre = True
    if found_genre:
        for song in filtered_songs:
            print(f"{song['title']} - {song['artist']}")
    else:
        print("Genre not found!")


def sort_by_rating(playlist):
    return sorted(playlist, key=lambda s: s["rating"], reverse=True)

def sort_by_duration(playlist):
    return sorted(playlist, key=lambda s: s["duration"])

def total_time(playlist):
    total = 0
    for song in playlist:
        total += song["duration"]
    return total

def mood_based(playlist, mood):
    if mood == "party":
        wanted = ["Pop", "Dance", "Hip-Hop"]
    elif mood == "study":
        wanted = ["Classical", "Jazz"]
    elif mood == "relax":
        wanted = ["Jazz", "Ambient"]
    else:
        wanted = []
    
    for song in playlist:
        if song["genre"] in wanted:
            print(f"{song['title']} - {song['artist']} ({song['genre']})")


def menu():
    while True:
        print("\n=== Smart Playlist Manager ===")
        print("\n1. Show songs")
        print("2. Filter by genre")
        print("3. Sort by rating")
        print("4. Sort by duration")
        print("5. Total time")
        print("6. Mood-based recommendation")
        print("0. Exit")
        
        choice = input("\nChoose an option: ")
        
        if choice == "1":
            show_songs(playlist)
        
        elif choice == "2":
            print("\nAvailable genres:")

            genres = set()
            for song in playlist:
                genres.add(song["genre"])

            for g in genres:
                print("-", g)
            
            g = input("\nChoose a genre: ")
            filter_by_genre(playlist, g)
        
        elif choice == "3":  # Sort by rating
            sorted_list = sorted(
                playlist,
                key=lambda s: (-s["rating"], s["title"])  # negative rating = descending
            )
            for song in sorted_list:
                print(song["title"], "-", song["artist"], "(rating:", song["rating"], ")")

        elif choice == "4":
            sorted_list_dur = sort_by_duration(playlist)
            show_songs(sorted_list_dur)

        elif choice == "5":
            print("Total duration:", total_time(playlist), "seconds")

        elif choice == "6":
            mood = input("Mood (party/study/relax): ")
            mood_based(playlist, mood)
            
        elif choice == "0":
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid option.")

menu()