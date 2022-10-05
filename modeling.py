import pandas as pd
import numpy as np
import requests
import re
import string

class Modeling_rule_based:
    # get product name manual
    ## filter product name, perlu di regex lagi
    def get_product(self, p):
        p = p[p.find('*'):len(p)]
        p = p[:p.find('\n')]
        p = p.lower()
        p = re.sub("[^0-9a-z]+", " ", p)

        replacements = {
            '(?<=\s|[0-9])+(g|gr)+(?=\s|\.|$)' : 'gram',
            '(?<=\s|[0-9])+mg+(?=\s|\.|$)' : 'miligram',
            '(?<=\s|[0-9])+ml+(?=\s|\.|$)' : 'mililiter',
            '(?<=\s|[0-9])+l+(?=\s|\.|$)' : 'liter'
        }

        for key, value in replacements.items():
            p = re.sub(key, value, p)

        # p = re.sub("gr",  "gram", p)
        # p = re.sub("ml", "mililiter", p)
        return p

    # create question manual
    ## keyword berdasarkan huruf terbanyak yang muncul
    def create_questions(self, c, q):
        list_q = []
        if c == "product_info":
            q1 = q 
            q2 = "kegunaan " + q
            q3 = "penggunaan " + q
            q4 = "cara pakai " + q
            q5 = "bagaimana menggunakan " + q
            q6 = "cara menggunakan " + q
            q7 = "cara penyimpanan " + q
            q8 = "bagaimana menyimpan " + q
            q9 = "cara menyimpan " + q
            q10 = "apa itu " + q
            q11 = "harga " + q
            q12 = "berapa biaya " + q
            list_q = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12]
        elif c == "treatment_info":
            q1 = q
            q2 = "apa itu " + q
            q3 = "manfaat " + q
            q4 = "manfaat menggunakan" + q
            q5 = "keuntungan menggunakan " + q
            q6 = "prosedur " + q
            q7 = "bagaimana pengerjaan " + q
            q8 = "bagaimana cara kerja " + q
            q9 = "durasi " + q
            q10 = "berapa lama " + q
            q11 = "harga " + q
            q12 = "berapa biaya " + q
            list_q = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12]
        
        return list_q