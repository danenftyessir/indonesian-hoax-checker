# Kamus singkatan/bahasa informal -> bentuk baku, satu token ke satu token.
# Dipakai normalisasi (informal -> baku) dan augmentasi (baku -> informal).
# Singkatan ambigu sengaja tidak dimasukkan, mis. "dr" (dokter/dari) dan "km" (kilometer/kamu).

SLANG_TO_FORMAL = {
    "yg": "yang", "dgn": "dengan", "dg": "dengan", "utk": "untuk", "tdk": "tidak", "gak": "tidak",
    "ga": "tidak", "gk": "tidak", "nggak": "tidak", "enggak": "tidak", "ngga": "tidak", "tak": "tidak",
    "sdh": "sudah", "udh": "sudah", "udah": "sudah", "blm": "belum", "krn": "karena", "karna": "karena",
    "jg": "juga", "aja": "saja", "aj": "saja", "sj": "saja", "bgt": "banget", "bs": "bisa", "bsa": "bisa",
    "dri": "dari", "pd": "pada", "kpd": "kepada", "sm": "sama", "sma": "sama",
    "spt": "seperti", "kyk": "kayak", "kalo": "kalau", "klo": "kalau", "kl": "kalau", "tp": "tapi",
    "tpi": "tapi", "trs": "terus", "lg": "lagi", "lgi": "lagi", "skrg": "sekarang", "skrng": "sekarang",
    "bsk": "besok", "besuk": "besok", "kmrn": "kemarin", "kemaren": "kemarin", "hr": "hari",
    "bln": "bulan", "thn": "tahun", "tgl": "tanggal", "jt": "juta", "rb": "ribu", "org": "orang",
    "msh": "masih", "mau": "mau", "mo": "mau", "gimana": "bagaimana", "gmn": "bagaimana",
    "knp": "kenapa", "napa": "kenapa", "apa2": "apa-apa", "sy": "saya", "gw": "saya", "gue": "saya",
    "aku": "aku", "lu": "kamu", "lo": "kamu", "kita2": "kita", "pemprov": "pemprov",
    "pemda": "pemda", "pemkot": "pemkot", "dpt": "dapat", "dapet": "dapat", "bkn": "bukan",
    "jgn": "jangan", "hrs": "harus", "smua": "semua", "sdgkan": "sedangkan", "sbg": "sebagai",
    "tsb": "tersebut", "dll": "dll", "dsb": "dsb", "ttg": "tentang", "stlh": "setelah", "sblm": "sebelum",
    "gaes": "gaes", "guys": "guys",
}
# Hapus entri yang tidak berubah supaya kamus balik bersih
SLANG_TO_FORMAL = {k: v for k, v in SLANG_TO_FORMAL.items() if k != v}

# Kamus balik untuk augmentasi: satu bentuk informal per kata baku (yang pertama di atas)
FORMAL_TO_SLANG = {}
for slang, formal in SLANG_TO_FORMAL.items():
    FORMAL_TO_SLANG.setdefault(formal, slang)
