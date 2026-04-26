ore_vizionate = 5
gen_preferat = "action"


if ore_vizionate > 10:
    # Utilizator foarte activ
    if gen_preferat == "action":
        recomandare = "Top 10 filme de acțiune exclusive"
    elif gen_preferat == "comedy":
        recomandare = "Stand-up comedy specials noi"
    else:
        recomandare = "Seriale premiate recent"
elif ore_vizionate > 3:
    recomandare = f"Continuă să vezi action"
else:
    recomandare = "Top 5 trending acum"


print(recomandare)

# Integrare baza date (.txt/md/sqlite)

