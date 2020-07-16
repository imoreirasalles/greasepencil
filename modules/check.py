import pandas as pd

table = pd.read_csv("./Firmo_lote_1.csv", names=["id", "issue", "obs"])

table["crop"] = table["issue"].str.findall(
    r"(?P<crop>[\w-]+)(?=(?:\s*,\s*[\w-]+)*\s+refazer crop)"
)
table["missing"] = table["issue"].str.findall(
    r"(?!Faltam? imagem?n?s?) (?P<missing>\d{2})"
)
table = table.explode("crop")
table = table.explode("missing")

table["missing"] = table["id"] + "-" + table["missing"]
crop = table["crop"].dropna()
missing = table["missing"].dropna()

crop.to_csv("./crop1.csv", index=False)
table.to_csv("./table1.csv")
missing.to_csv("./missing1.csv", index=False)

# pd.set_option("display.max_rows", None)
# print(crop)
