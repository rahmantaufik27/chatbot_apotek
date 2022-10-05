import re

"""
    1. replace gr/g => gram, mg => miligram, l => liter, ml => mililiter
    2. replace unit jika sebelum unit terdapat angka/spasi dan setelah unit terdapat spasi/titik/end.
    3. tidak mengganti unit pada awal kalimat

    contoh ada pada fungsi main
"""

def unit_transform(str):
    replacements = {
            '(?<=\s|[0-9])+(g|gr)+(?=\s|\.|$)' : 'gram',
            '(?<=\s|[0-9])+mg+(?=\s|\.|$)' : 'miligram',
            '(?<=\s|[0-9])+ml+(?=\s|\.|$)' : 'mililiter',
            '(?<=\s|[0-9])+l+(?=\s|\.|$)' : 'liter'
        }

    for key, value in replacements.items():
        str = re.sub(key, value, str)
    return (str)


def main():    
    x = "g Berikut melalui 4mg 5 gr 5l l Informasig l gmengenai lalu *ACNE TONER 2 (AT 2)* 60 g. asdAdasd g"
    print(unit_transform(x))


if __name__ == "__main__":
    main()
