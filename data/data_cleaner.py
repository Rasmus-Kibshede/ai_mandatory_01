from pandas import DataFrame
import difflib


def clean(data: DataFrame):
    data.dropna(inplace=True)
    data.fillna(0, inplace=True)
    data.drop_duplicates(inplace=True)
    data.replace("'", "", regex=True)
    spell_checker(data)


def spell_checker(data: DataFrame):
    for i, entry in enumerate(data['make']):
        if isinstance(entry, str):
            corrected_make = correct_make(entry.strip())
            data.at[i, 'make'] = corrected_make

    data.to_csv('data/modified_dataset.csv', index=False)


def correct_make(make):
    corrected_make = [
        'chevrolet', 'buick', 'plymouth', 'amc', 'ford', 'pontiac', 'dodge', 'toyota',
        'datsun', 'volkswagen', 'peugeot', 'audi', 'saab', 'bmw', 'hi', 'chevy',
        'chrysler', 'mazda', 'opel', 'fiat', 'mercury', 'volvo', 'renault', 'subaru',
        'capri', 'mercedes-benz', 'cadillac', 'vw'
    ]

    closest_match = difflib.get_close_matches(make, corrected_make, n=1, cutoff=0.6)
    if closest_match:
        if closest_match[0] == 'vw':
            return 'volkswagen'
        if closest_match[0] == 'chevy':
            return 'chevrolet'
        return closest_match[0]
    else:
        return make
