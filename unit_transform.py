import re

"""
Unit Transform
    1. replace gr/g => gram, mg => miligram, l => liter, ml => mililiter
    2. replace unit jika sebelum unit terdapat angka/spasi dan setelah unit terdapat spasi/titik/end.
    3. tidak mengganti unit pada awal kalimat

Product Name Transform
    1. menambahkan spasi pada nama produk seperti 'erha4' menjadi 'erha 4'
    
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

def product_name_transform(str):
    str = re.sub("(?<=[A-Za-z])+[0-9]+(?=\s|\.|$)",
                 lambda group: " "+group[0], str)
    return str

def main():
    x = "erha4 g erha4 Berikut er4h 4rh melalui 4mg 5 gr 5l l Informasig l gmengenai lalu *ACNE TONER 2 (AT 2)* 60 g. asdAdasd g erha4"
    print(x)
    print(product_name_transform(x))


if __name__ == "__main__":
    main()
