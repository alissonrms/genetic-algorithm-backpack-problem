import csv
from pathlib import Path

def get_project_root() -> Path:
    print(Path(__file__).parent.parent)
    return Path(__file__).parent.parent

def readData(fileName):
    items = []
    filePath = f'{get_project_root()}/assets/{fileName}'
    with open(filePath, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            peso = float(row['Peso'].replace(',', '.'))
            utilidade = int(row['Utilidade'])
            item = (peso, utilidade)
            items.append(item)

    return items
