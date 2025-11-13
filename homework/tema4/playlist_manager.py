# I. Smart Playlist Manager

# Listă cu melodii (dicționare: titlu, artist, gen, durata, rating).
# Operații: afișare, filtrare după gen, sortare după rating/durată, calcul timp total.
# Algoritm „mood-based”: utilizatorul alege dispoziția („party”, „study”, „relax”) → programul recomandă melodii potrivite.


playlist = [
    {"title": "Highest In The Room", "artist": "Travis Scott", "genre": "Electronica", "duration": 177, "rating": 7},
    {"title": "Godzilla", "artist": "Eminem", "genre": "Hip-Hop", "duration": 267, "rating": 8},
    {"title": "Love Story", "artist": "Taylor Swift", "genre": "Country", "duration": 237, "rating": 9},
    {"title": "Let Me Down Slowly", "artist": "Alec Benjamin", "genre": "Alternative", "duration": 178, "rating": 8},
    {"title": "Fly Me to the Moon", "artist": "Frank Sinatra", "genre": "Jazz", "duration":148 , "rating": 8},
    {"title": "Symphony No. 5", "artist": "Beethoven", "genre": "Classical", "duration": 2880, "rating": 10}
]


# --- Menu loop ---
while True:
    print("\nSmart Playlist Manager")
    print("1. Show songs")
    print("2. Filter by genre")
    print("3. Sort by rating")
    print("4. Sort by duration")
    print("5. Total time")
    print("6. Mood-based recommendation")
    print("7. Add/Delete songs")
    print("0. Exit")

    choice = input("\nChoose an option from the menu above: ")

    if choice == "1":  # Show songs
        for song in playlist:
            print(song["title"], "-", song["artist"], "(", song["genre"], ")", 
                  str(song["duration"]) + "s,", "rating:", song["rating"])

    elif choice == "2":  # Filter by genre
        print("\nAvailable genres:")
        for song in playlist:
            print("-", song["genre"])
    
        g = input("\nChoose a genre: ")
        for song in playlist:
            if song["genre"].lower() == g.lower():
                print(song["title"], "-", song["artist"], "(", song["genre"], ")")

    elif choice == "3":  # Sort by rating
        sorted_list = sorted(playlist, key=lambda s: s["rating"], reverse=True)
        for song in sorted_list:
            print(song["title"], "-", song["artist"], "(rating:", song["rating"], ")")

    elif choice == "4":  # Sort by duration
        sorted_list = sorted(playlist, key=lambda s: s["duration"])
        for song in sorted_list:
            print(song["title"], "-", song["artist"], "(", str(song["duration"]) + "s )")

    elif choice == "5":  # Total duration
        total = 0
        for song in playlist:
            total += song["duration"]
        print("\nTotal duration:", total, "seconds")

    elif choice == "6":  # Mood-based
        mood = input("Mood (party/study/relax/comfort): ")
        if mood == "party":
            wanted = ["Electronica", "Hip-Hop"]
        elif mood == "study":
            wanted = ["Classical", "Jazz"]
        elif mood == "comfort":
            wanted = ["Country"]
        elif mood == "relax":
            wanted = ["Jazz", "Alternative"]
        else:
            wanted = []

        for song in playlist:
            if song["genre"] in wanted:
                print(song["title"], "-", song["artist"], "(", song["genre"], ")")

    elif choice == "7":  # Add/Delete songs
        sub = input("Type 'add' to add a song or 'delete' to remove one: ")

        if sub.lower() == "add":
            title = input("Title: ")
            artist = input("Artist: ")
            genre = input("Genre: ")
            duration = int(input("Duration (seconds): "))
            rating = int(input("Rating (1-10): "))
            new_song = {"title": title, "artist": artist, "genre": genre,
                        "duration": duration, "rating": rating}
            playlist.append(new_song)
            print("Song added!")

        elif sub.lower() == "delete":
            title = input("Enter the title of the song to delete: ")
            found = False
            for song in playlist:
                if song["title"].lower() == title.lower():
                    playlist.remove(song)
                    print("Song deleted!")
                    found = True
                    break
            if not found:
                print("Song not found.")

        else:
            print("Invalid choice.")

    elif choice == "0":  # Exit
        break

    else:
        print("Invalid option.")