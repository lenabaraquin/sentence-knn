# Entrée : fichier texte brut
# Sortie : fichier json (clés = ["id", "text"])
# Sépare le texte du fichier d'entrée en phrases (chaines séparées par un point) et envoie chaque phrase dans une entrée séparée du fichier json de sortie
BEGIN {
  print "["
}
{
  ORS = "" # Bricolage bis
  n = split($0, a, /\.[ \t]*/)
  for (i = 1; i <= n; i++) {
    if (length(a[i]) > 0) {
      if (i != 1) print ",\n" # Bricolage pour ne pas intégrer le séparateur "," à la dernière ligne du fichier
        else print ""
      gsub(/"/, "\\\"", a[i])
      printf "  \{\"id\": \"%d\"\, \"text\": \"%s\"\}", ++c, a[i]
    }
  }
}
END {
  print "\n]"
}
