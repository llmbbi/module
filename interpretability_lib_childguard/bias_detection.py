
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional

class BiasDetector:
    """
    Detects and analyzes demographic biases in feature attributions.
    
    Extended to support before/after fine-tuning comparisons to answer:
    - Does fine-tuning amplify demographic biases present in base models?
    - Does task-specific training reduce spurious correlations?
    """
    
    def __init__(self):
        # Multilingual demographic token groups
        # Covers the 21 M-ABSA languages: ar, da, de, en, es, fr, hi, hr, id,
        # ja, ko, nl, pt, ru, sk, sv, sw, th, tr, vi, zh
        self.demographic_groups = {
            "gender": {
                "male": [
                    # English
                    "he", "him", "his", "man", "male", "boy", "father", "son", "husband", "brother",
                    # Arabic
                    "هو", "رجل", "ذكر", "ولد", "أب", "ابن", "زوج", "أخ",
                    # Danish
                    "han", "mand", "dreng", "far", "søn", "bror", "ægtemand",
                    # German
                    "er", "ihm", "sein", "mann", "männlich", "junge", "vater", "sohn", "ehemann", "bruder",
                    # Spanish
                    "él", "hombre", "masculino", "chico", "niño", "padre", "hijo", "esposo", "hermano", "varón",
                    # French
                    "il", "lui", "homme", "masculin", "garçon", "père", "fils", "mari", "frère",
                    # Hindi
                    "वह", "उसका", "आदमी", "पुरुष", "लड़का", "पिता", "बेटा", "पति", "भाई",
                    # Croatian
                    "on", "njega", "muškarac", "muško", "dječak", "otac", "sin", "muž", "brat",
                    # Indonesian
                    "dia", "laki-laki", "pria", "ayah", "putra", "suami", "saudara",
                    # Japanese
                    "彼", "男", "男性", "少年", "父", "息子", "夫", "兄弟",
                    # Korean
                    "그", "남자", "남성", "소년", "아버지", "아들", "남편", "형제", "오빠", "형",
                    # Dutch
                    "hij", "hem", "zijn", "man", "mannelijk", "jongen", "vader", "zoon", "echtgenoot", "broer",
                    # Portuguese
                    "ele", "homem", "masculino", "menino", "pai", "filho", "marido", "irmão",
                    # Russian
                    "он", "его", "мужчина", "мужской", "мальчик", "отец", "сын", "муж", "брат",
                    # Slovak
                    "muž", "chlapec", "otec", "syn", "manžel", "brat",
                    # Swedish
                    "han", "man", "pojke", "far", "pappa", "son", "make", "bror",
                    # Swahili
                    "yeye", "mtu", "mvulana", "baba", "mwana", "mume", "kaka", "ndugu",
                    # Thai
                    "เขา", "ผู้ชาย", "ชาย", "เด็กชาย", "พ่อ", "ลูกชาย", "สามี", "พี่ชาย",
                    # Turkish
                    "erkek", "adam", "oğlan", "baba", "oğul", "koca", "kardeş", "abi",
                    # Vietnamese
                    "anh", "ông", "đàn ông", "nam", "cha", "bố", "chồng", "anh trai",
                    # Chinese
                    "他", "男人", "男性", "男孩", "父亲", "儿子", "丈夫", "兄弟", "爸爸",
                ],
                "female": [
                    # English
                    "she", "her", "hers", "woman", "female", "girl", "mother", "daughter", "wife", "sister",
                    # Arabic
                    "هي", "امرأة", "أنثى", "بنت", "أم", "ابنة", "زوجة", "أخت",
                    # Danish
                    "hun", "kvinde", "pige", "mor", "datter", "søster", "hustru",
                    # German
                    "sie", "ihr", "frau", "weiblich", "mädchen", "mutter", "tochter", "ehefrau", "schwester",
                    # Spanish
                    "ella", "mujer", "femenino", "chica", "niña", "madre", "hija", "esposa", "hermana",
                    # French
                    "elle", "femme", "féminin", "fille", "mère", "épouse", "sœur",
                    # Hindi
                    "वह", "उसकी", "औरत", "महिला", "लड़की", "माँ", "बेटी", "पत्नी", "बहन",
                    # Croatian
                    "ona", "žena", "žensko", "djevojka", "majka", "kći", "supruga", "sestra",
                    # Indonesian
                    "perempuan", "wanita", "ibu", "putri", "istri", "saudari",
                    # Japanese
                    "彼女", "女", "女性", "少女", "母", "娘", "妻", "姉妹",
                    # Korean
                    "그녀", "여자", "여성", "소녀", "어머니", "딸", "아내", "자매", "언니", "누나",
                    # Dutch
                    "zij", "haar", "vrouw", "vrouwelijk", "meisje", "moeder", "dochter", "echtgenote", "zus",
                    # Portuguese
                    "ela", "mulher", "feminino", "menina", "mãe", "filha", "esposa", "irmã",
                    # Russian
                    "она", "её", "женщина", "женский", "девочка", "мать", "дочь", "жена", "сестра",
                    # Slovak
                    "žena", "dievča", "matka", "dcéra", "manželka", "sestra",
                    # Swedish
                    "hon", "kvinna", "flicka", "mor", "mamma", "dotter", "maka", "syster",
                    # Swahili
                    "mwanamke", "msichana", "mama", "binti", "mke", "dada",
                    # Thai
                    "เธอ", "ผู้หญิง", "หญิง", "เด็กหญิง", "แม่", "ลูกสาว", "ภรรยา", "พี่สาว",
                    # Turkish
                    "kadın", "kız", "anne", "kızı", "karı", "abla",
                    # Vietnamese
                    "chị", "bà", "phụ nữ", "nữ", "mẹ", "vợ", "chị gái",
                    # Chinese
                    "她", "女人", "女性", "女孩", "母亲", "女儿", "妻子", "姐妹", "妈妈",
                ],
            },
            "age": {
                "young": [
                    # English
                    "young", "youth", "teenager", "teen", "child", "kid", "boy", "girl",
                    # Arabic
                    "شاب", "شباب", "مراهق", "طفل",
                    # Danish
                    "ung", "ungdom", "teenager", "barn",
                    # German
                    "jung", "jugend", "teenager", "kind",
                    # Spanish
                    "joven", "juventud", "adolescente", "niño", "niña",
                    # French
                    "jeune", "jeunesse", "adolescent", "enfant",
                    # Hindi
                    "जवान", "युवा", "किशोर", "बच्चा",
                    # Croatian
                    "mlad", "mladost", "tinejdžer", "dijete",
                    # Indonesian
                    "muda", "remaja", "anak",
                    # Japanese
                    "若い", "若者", "青年", "子供", "少年", "少女",
                    # Korean
                    "젊은", "청년", "청소년", "아이", "어린이",
                    # Dutch
                    "jong", "jeugd", "tiener", "kind",
                    # Portuguese
                    "jovem", "juventude", "adolescente", "criança",
                    # Russian
                    "молодой", "молодёжь", "подросток", "ребёнок",
                    # Slovak
                    "mladý", "mládež", "tínedžer", "dieťa",
                    # Swedish
                    "ung", "ungdom", "tonåring", "barn",
                    # Swahili
                    "kijana", "vijana", "mtoto",
                    # Thai
                    "หนุ่ม", "สาว", "เยาวชน", "วัยรุ่น", "เด็ก",
                    # Turkish
                    "genç", "gençlik", "çocuk",
                    # Vietnamese
                    "trẻ", "thanh niên", "thiếu niên", "trẻ em",
                    # Chinese
                    "年轻", "青年", "少年", "孩子", "儿童",
                ],
                "old": [
                    # English
                    "old", "elderly", "senior", "aged", "mature", "adult", "grandpa", "grandma",
                    # Arabic
                    "كبير", "مسن", "عجوز", "جد", "جدة",
                    # Danish
                    "gammel", "ældre", "bedstefar", "bedstemor",
                    # German
                    "alt", "älter", "ältere", "opa", "oma", "großvater", "großmutter",
                    # Spanish
                    "viejo", "anciano", "mayor", "abuelo", "abuela",
                    # French
                    "vieux", "âgé", "ancien", "aîné", "grand-père", "grand-mère",
                    # Hindi
                    "बूढ़ा", "वृद्ध", "बुजुर्ग", "दादा", "दादी",
                    # Croatian
                    "star", "stariji", "djed", "baka",
                    # Indonesian
                    "tua", "lansia", "kakek", "nenek",
                    # Japanese
                    "老人", "高齢", "年配", "祖父", "祖母", "おじいさん", "おばあさん",
                    # Korean
                    "늙은", "노인", "어르신", "할아버지", "할머니",
                    # Dutch
                    "oud", "ouderen", "bejaarde", "opa", "oma",
                    # Portuguese
                    "velho", "idoso", "avô", "avó",
                    # Russian
                    "старый", "пожилой", "дедушка", "бабушка",
                    # Slovak
                    "starý", "starší", "dedko", "babka",
                    # Swedish
                    "gammal", "äldre", "farfar", "mormor",
                    # Swahili
                    "mzee", "bibi", "babu",
                    # Thai
                    "แก่", "ผู้สูงอายุ", "คนชรา", "ปู่", "ย่า",
                    # Turkish
                    "yaşlı", "büyük", "dede", "nine", "büyükanne", "büyükbaba",
                    # Vietnamese
                    "già", "người già", "ông", "bà",
                    # Chinese
                    "老", "老人", "老年", "年老", "爷爷", "奶奶", "祖父", "祖母",
                ],
            },
            "race": {
                "white": [
                    # English
                    "white", "caucasian", "european", "american",
                    # Arabic
                    "أبيض", "أوروبي", "قوقازي",
                    # German
                    "weiß", "europäisch", "kaukasisch",
                    # Spanish
                    "blanco", "europeo", "caucásico",
                    # French
                    "blanc", "européen", "caucasien",
                    # Hindi
                    "गोरा", "यूरोपीय",
                    # Japanese
                    "白人", "ヨーロッパ人",
                    # Korean
                    "백인", "유럽인",
                    # Dutch
                    "wit", "europees", "blank",
                    # Portuguese
                    "branco", "europeu", "caucasiano",
                    # Russian
                    "белый", "европеец", "кавказец",
                    # Swedish
                    "vit", "europeisk",
                    # Thai
                    "คนขาว", "ยุโรป",
                    # Turkish
                    "beyaz", "avrupalı",
                    # Vietnamese
                    "da trắng", "người châu âu",
                    # Chinese
                    "白人", "欧洲人", "白种人",
                ],
                "black": [
                    # English
                    "black", "african", "afro",
                    # Arabic
                    "أسود", "أفريقي",
                    # German
                    "schwarz", "afrikanisch",
                    # Spanish
                    "negro", "africano", "afro",
                    # French
                    "noir", "africain", "afro",
                    # Hindi
                    "काला", "अफ्रीकी",
                    # Japanese
                    "黒人", "アフリカ人",
                    # Korean
                    "흑인", "아프리카인",
                    # Dutch
                    "zwart", "afrikaans",
                    # Portuguese
                    "negro", "preto", "africano", "afro",
                    # Russian
                    "чёрный", "африканец",
                    # Swedish
                    "svart", "afrikansk",
                    # Thai
                    "คนดำ", "แอฟริกัน",
                    # Turkish
                    "siyah", "afrikalı",
                    # Vietnamese
                    "da đen", "người châu phi",
                    # Chinese
                    "黑人", "非洲人",
                ],
            },
            "religion": {
                "christian": [
                    # English
                    "christian", "church", "bible", "jesus", "priest",
                    # Arabic
                    "مسيحي", "كنيسة", "إنجيل", "يسوع", "قسيس",
                    # German
                    "christlich", "kirche", "bibel", "priester",
                    # Spanish
                    "cristiano", "iglesia", "biblia", "sacerdote",
                    # French
                    "chrétien", "église", "prêtre",
                    # Hindi
                    "ईसाई", "गिरजाघर", "बाइबिल", "पादरी",
                    # Japanese
                    "キリスト教", "教会", "聖書", "牧師",
                    # Korean
                    "기독교", "교회", "성경", "목사",
                    # Dutch
                    "christelijk", "kerk", "bijbel", "priester",
                    # Portuguese
                    "cristão", "igreja", "bíblia", "padre",
                    # Russian
                    "христианин", "церковь", "библия", "священник",
                    # Swedish
                    "kristen", "kyrka", "bibel", "präst",
                    # Thai
                    "คริสเตียน", "โบสถ์", "พระคัมภีร์",
                    # Turkish
                    "hristiyan", "kilise", "incil", "papaz",
                    # Vietnamese
                    "thiên chúa giáo", "nhà thờ", "kinh thánh",
                    # Chinese
                    "基督教", "教堂", "圣经", "牧师",
                ],
                "muslim": [
                    # English
                    "muslim", "islam", "mosque", "quran", "allah", "hijab",
                    # Arabic
                    "مسلم", "إسلام", "مسجد", "قرآن", "الله", "حجاب",
                    # German
                    "muslimisch", "moschee", "koran", "kopftuch",
                    # Spanish
                    "musulmán", "mezquita", "corán",
                    # French
                    "musulman", "mosquée", "coran", "voile",
                    # Hindi
                    "मुस्लिम", "इस्लाम", "मस्जिद", "कुरान", "हिजाब",
                    # Japanese
                    "イスラム", "モスク", "コーラン", "ヒジャブ",
                    # Korean
                    "무슬림", "이슬람", "모스크", "코란", "히잡",
                    # Dutch
                    "moslim", "moskee", "koran", "hoofddoek",
                    # Portuguese
                    "muçulmano", "mesquita", "alcorão",
                    # Russian
                    "мусульманин", "ислам", "мечеть", "коран", "хиджаб",
                    # Swedish
                    "muslim", "moské", "koranen", "slöja",
                    # Thai
                    "มุสลิม", "อิสลาม", "มัสยิด", "อัลกุรอาน", "ฮิญาบ",
                    # Turkish
                    "müslüman", "cami", "kuran",
                    # Vietnamese
                    "hồi giáo", "nhà thờ hồi giáo",
                    # Chinese
                    "穆斯林", "伊斯兰", "清真寺", "古兰经",
                ],
            },
        }
        # Multilingual sentiment-bearing attribute tokens for WAT
        self.sentiment_attributes = {
            "positive": [
                # English
                "good", "great", "excellent", "happy", "love", "wonderful", "best", "positive",
                # Arabic
                "جيد", "ممتاز", "سعيد", "حب", "رائع", "أفضل",
                # German
                "gut", "großartig", "ausgezeichnet", "glücklich", "liebe", "wunderbar", "beste",
                # Spanish
                "bueno", "genial", "excelente", "feliz", "amor", "maravilloso", "mejor",
                # French
                "bon", "super", "excellent", "heureux", "amour", "merveilleux", "meilleur",
                # Hindi
                "अच्छा", "बढ़िया", "उत्कृष्ट", "खुश", "प्यार", "सर्वश्रेष्ठ",
                # Japanese
                "良い", "素晴らしい", "最高", "幸せ", "愛", "優秀",
                # Korean
                "좋은", "훌륭한", "최고", "행복", "사랑", "우수한",
                # Dutch
                "goed", "geweldig", "uitstekend", "blij", "liefde", "beste",
                # Portuguese
                "bom", "ótimo", "excelente", "feliz", "amor", "maravilhoso", "melhor",
                # Russian
                "хороший", "отличный", "превосходный", "счастливый", "любовь", "лучший",
                # Thai
                "ดี", "ยอดเยี่ยม", "สุดยอด", "มีความสุข", "รัก",
                # Turkish
                "iyi", "harika", "mükemmel", "mutlu", "sevgi", "en iyi",
                # Vietnamese
                "tốt", "tuyệt vời", "xuất sắc", "hạnh phúc", "yêu",
                # Chinese
                "好", "很好", "优秀", "开心", "爱", "最好",
            ],
            "negative": [
                # English
                "bad", "terrible", "awful", "sad", "hate", "worst", "horrible", "negative",
                # Arabic
                "سيء", "فظيع", "حزين", "كراهية", "أسوأ",
                # German
                "schlecht", "schrecklich", "furchtbar", "traurig", "hass", "schlimmste",
                # Spanish
                "malo", "terrible", "horrible", "triste", "odio", "peor",
                # French
                "mauvais", "terrible", "affreux", "triste", "haine", "pire", "horrible",
                # Hindi
                "बुरा", "भयानक", "दुखी", "नफरत", "सबसे बुरा",
                # Japanese
                "悪い", "ひどい", "最悪", "悲しい", "嫌い",
                # Korean
                "나쁜", "끔찍한", "최악", "슬픈", "싫어",
                # Dutch
                "slecht", "verschrikkelijk", "vreselijk", "verdrietig", "haat", "slechtste",
                # Portuguese
                "mau", "ruim", "terrível", "horrível", "triste", "ódio", "pior",
                # Russian
                "плохой", "ужасный", "грустный", "ненависть", "худший",
                # Thai
                "แย่", "เลว", "น่ากลัว", "เศร้า", "เกลียด",
                # Turkish
                "kötü", "korkunç", "berbat", "üzgün", "nefret", "en kötü",
                # Vietnamese
                "xấu", "tệ", "kinh khủng", "buồn", "ghét",
                # Chinese
                "坏", "糟糕", "可怕", "难过", "恨", "最差",
            ],
        }

    def compute_attribution_mass(self, attribution_list, token_set):
        """
        Calculates the total attribution mass for a specific set of tokens in a single explanation.
        attribution_list: List of (word, score) tuples.
        token_set: Set of target tokens (strings) to aggregate mass for.
        """
        mass = 0.0
        normalized_set = set(t.lower() for t in token_set)
        
        for word, score in attribution_list:
            # Simple containment check; could be regex or exact match
            # "man" should match "man" or "man's"? We'll assume simple match for now
            if word.lower() in normalized_set:
                mass += abs(score) # Use absolute magnitude, or raw? Usually mass implies magnitude.
        return mass

    def analyze_bias(self, attributions_batch, group_a_tokens, group_b_tokens):
        """
        Performs statistical comparison between two demographic groups.
        attributions_batch: List of explanations (each explanation is a list of (word, score)).
        group_a_tokens: List of tokens for Group A.
        group_b_tokens: List of tokens for Group B.
        """
        masses_a = []
        masses_b = []
        
        for attr in attributions_batch:
            mass_a = self.compute_attribution_mass(attr, group_a_tokens)
            mass_b = self.compute_attribution_mass(attr, group_b_tokens)
            masses_a.append(mass_a)
            masses_b.append(mass_b)
            
        masses_a = np.array(masses_a)
        masses_b = np.array(masses_b)
        
        # Statistical Test: Independent T-test (assuming different samples, or we treat them as paired?)
        # Since we are comparing masses *within the same samples* (e.g. how much attention to 'he' vs 'she' in the same text?),
        # or across a dataset where these words appear?
        # Usually, bias is: "Does the model attend more to male words than female words overall?"
        # Paired T-test is appropriate if we are comparing A vs B weights in the SAME documents.
        # But if the documents are different (some have 'he', some 'she'), independent is better.
        # However, here we just sum mass. If a document has NO 'he' words, mass is 0.
        # So we are comparing the distributions of "Attention to A" vs "Attention to B" over the dataset.
        
        # We'll use a Paired T-test (rel) if we care about the difference per sample, 
        # but T-test independent (ind) is safer if they are not strictly paired. 
        # Actually, `scipy.stats.ttest_rel` checks if the mean difference is zero.
        # `scipy.stats.ttest_ind` checks if the means of two independent samples are different.
        # Let's use `ttest_ind` as a general approach.
        
        # Statistical Test: Independent T-test
        # Handle cases with no mass (avoid NaN)
        if np.all(masses_a == 0) and np.all(masses_b == 0):
            t_stat, p_val = 0.0, 1.0
            cohen_d = 0.0
        elif np.std(masses_a) == 0 and np.std(masses_b) == 0:
            if np.mean(masses_a) == np.mean(masses_b):
                t_stat, p_val = 0.0, 1.0
                cohen_d = 0.0
            else:
                t_stat, p_val = (np.inf if np.mean(masses_a) > np.mean(masses_b) else -np.inf), 0.0
                cohen_d = np.inf
        else:
            try:
                t_stat, p_val = stats.ttest_ind(masses_a, masses_b, equal_var=False)
                
                # Cohen's d = (mean1 - mean2) / pooled_std
                n1, n2 = len(masses_a), len(masses_b)
                v1, v2 = np.var(masses_a, ddof=1), np.var(masses_b, ddof=1)
                pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
                if pooled_std > 0:
                    cohen_d = (np.mean(masses_a) - np.mean(masses_b)) / pooled_std
                else:
                    cohen_d = 0.0
                    
                if np.isnan(t_stat):
                    t_stat, p_val, cohen_d = 0.0, 1.0, 0.0
            except:
                t_stat, p_val, cohen_d = 0.0, 1.0, 0.0
        
        # 95% Confidence Interval for the difference in means
        mean_diff = np.mean(masses_a) - np.mean(masses_b)
        se_diff = np.sqrt(np.var(masses_a, ddof=1)/len(masses_a) + np.var(masses_b, ddof=1)/len(masses_b)) if len(masses_a) > 0 and len(masses_b) > 0 else 0
        ci_low = mean_diff - 1.96 * se_diff
        ci_high = mean_diff + 1.96 * se_diff

        return {
            "mean_mass_a": float(np.mean(masses_a)),
            "mean_mass_b": float(np.mean(masses_b)),
            "std_mass_a": float(np.std(masses_a)),
            "std_mass_b": float(np.std(masses_b)),
            "mean_difference": float(mean_diff),
            "cohen_d": float(cohen_d),
            "ci_95": [float(ci_low), float(ci_high)],
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant_bias": bool(p_val < 0.05)
        }

    def compute_wat_score(self, attributions_batch: List[List[Tuple[str, float]]], group_tokens: List[str]) -> Dict[str, float]:
        """
        Word Attribution Association Test (WAT) proxy for attributions.
        Measures if demographic tokens are more associated with positive or negative sentiment tokens.
        
        association = Mean(Attribution of demographic token when positive tokens are high) - 
                      Mean(Attribution of demographic token when negative tokens are high)
        """
        pos_attr_tokens = set(self.sentiment_attributes["positive"])
        neg_attr_tokens = set(self.sentiment_attributes["negative"])
        group_tokens_set = set(t.lower() for t in group_tokens)
        
        pos_associations = []
        neg_associations = []
        
        for attr in attributions_batch:
            # Find total mass of positive and negative attribute tokens
            pos_mass = self.compute_attribution_mass(attr, pos_attr_tokens)
            neg_mass = self.compute_attribution_mass(attr, neg_attr_tokens)
            group_mass = self.compute_attribution_mass(attr, group_tokens_set)
            
            if group_mass > 0:
                if pos_mass > neg_mass:
                    pos_associations.append(group_mass)
                elif neg_mass > pos_mass:
                    neg_associations.append(group_mass)
        
        mean_pos = np.mean(pos_associations) if pos_associations else 0.0
        mean_neg = np.mean(neg_associations) if neg_associations else 0.0
        
        return {
            "wat_score": float(mean_pos - mean_neg),
            "mean_pos_association": float(mean_pos),
            "mean_neg_association": float(mean_neg),
            "num_samples_with_group": len(pos_associations) + len(neg_associations)
        }

    # =========================================================================
    # NEW METHODS: Before/After Fine-tuning Bias Comparison
    # =========================================================================

    def compare_bias_phases(
        self,
        pre_bias_results: Dict[str, Any],
        post_bias_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare bias results between pre and post fine-tuning phases.
        
        Args:
            pre_bias_results: Bias analysis from before fine-tuning
            post_bias_results: Bias analysis from after fine-tuning
            
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {
            "pre_finetune": pre_bias_results,
            "post_finetune": post_bias_results
        }
        
        # Calculate changes
        pre_diff = pre_bias_results.get("mean_mass_a", 0) - pre_bias_results.get("mean_mass_b", 0)
        post_diff = post_bias_results.get("mean_mass_a", 0) - post_bias_results.get("mean_mass_b", 0)
        
        comparison["pre_mass_difference"] = float(pre_diff)
        comparison["post_mass_difference"] = float(post_diff)
        comparison["mass_difference_change"] = float(post_diff - pre_diff)
        
        # Calculate absolute bias magnitude
        pre_abs_diff = abs(pre_diff)
        post_abs_diff = abs(post_diff)
        comparison["pre_abs_bias"] = float(pre_abs_diff)
        comparison["post_abs_bias"] = float(post_abs_diff)
        
        # Determine if bias was amplified or reduced
        if pre_abs_diff > 0:
            change_ratio = (post_abs_diff - pre_abs_diff) / pre_abs_diff
            comparison["bias_change_ratio"] = float(change_ratio)
            
            if change_ratio > 0.1:
                comparison["bias_trend"] = "amplified"
            elif change_ratio < -0.1:
                comparison["bias_trend"] = "reduced"
            else:
                comparison["bias_trend"] = "stable"
        else:
            comparison["bias_change_ratio"] = float(post_abs_diff) if post_abs_diff > 0 else 0.0
            comparison["bias_trend"] = "emerged" if post_abs_diff > 0.1 else "stable"
        
        # Statistical significance comparison
        comparison["pre_significant"] = pre_bias_results.get("significant_bias", False)
        comparison["post_significant"] = post_bias_results.get("significant_bias", False)
        
        if comparison["pre_significant"] and not comparison["post_significant"]:
            comparison["significance_change"] = "bias_removed"
        elif not comparison["pre_significant"] and comparison["post_significant"]:
            comparison["significance_change"] = "bias_introduced"
        elif comparison["pre_significant"] and comparison["post_significant"]:
            comparison["significance_change"] = "bias_persists"
        else:
            comparison["significance_change"] = "no_significant_bias"
        
        return comparison

    def calculate_bias_amplification(
        self,
        pre_attributions: List[List[Tuple[str, float]]],
        post_attributions: List[List[Tuple[str, float]]],
        group_a_tokens: List[str],
        group_b_tokens: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate bias amplification metrics between phases.
        
        Args:
            pre_attributions: List of attribution lists from before fine-tuning
            post_attributions: List of attribution lists from after fine-tuning
            group_a_tokens: Tokens for demographic group A
            group_b_tokens: Tokens for demographic group B
            
        Returns:
            Dictionary with amplification metrics
        """
        # Analyze bias in both phases
        pre_bias = self.analyze_bias(pre_attributions, group_a_tokens, group_b_tokens)
        post_bias = self.analyze_bias(post_attributions, group_a_tokens, group_b_tokens)
        
        # Compare phases
        comparison = self.compare_bias_phases(pre_bias, post_bias)
        
        # Add per-sample analysis
        pre_diffs = []
        post_diffs = []
        
        for pre_attr, post_attr in zip(pre_attributions, post_attributions):
            pre_mass_a = self.compute_attribution_mass(pre_attr, group_a_tokens)
            pre_mass_b = self.compute_attribution_mass(pre_attr, group_b_tokens)
            post_mass_a = self.compute_attribution_mass(post_attr, group_a_tokens)
            post_mass_b = self.compute_attribution_mass(post_attr, group_b_tokens)
            
            pre_diffs.append(pre_mass_a - pre_mass_b)
            post_diffs.append(post_mass_a - post_mass_b)
        
        pre_diffs = np.array(pre_diffs)
        post_diffs = np.array(post_diffs)
        
        # Per-sample amplification
        if len(pre_diffs) > 0:
            amplification_per_sample = np.abs(post_diffs) - np.abs(pre_diffs)
            comparison["amplification_mean"] = float(np.mean(amplification_per_sample))
            comparison["amplification_std"] = float(np.std(amplification_per_sample))
            comparison["samples_with_increased_bias"] = int(np.sum(amplification_per_sample > 0))
            comparison["samples_with_decreased_bias"] = int(np.sum(amplification_per_sample < 0))
            comparison["samples_total"] = len(pre_diffs)
        
        return comparison

    def analyze_multiple_groups(
        self,
        attributions_batch: List[List[Tuple[str, float]]],
        groups: Optional[Dict[str, Dict[str, List[str]]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze bias across multiple demographic dimensions.
        
        Args:
            attributions_batch: List of attribution lists
            groups: Dictionary of demographic groups (default: use built-in groups)
            
        Returns:
            Dictionary with bias analysis for each dimension
        """
        if groups is None:
            groups = self.demographic_groups
        
        results = {}
        
        for dimension, group_tokens in groups.items():
            if len(group_tokens) >= 2:
                # Get first two groups for comparison
                group_names = list(group_tokens.keys())
                group_a_tokens = group_tokens[group_names[0]]
                group_b_tokens = group_tokens[group_names[1]]
                
                bias_result = self.analyze_bias(attributions_batch, group_a_tokens, group_b_tokens)
                bias_result["group_a_name"] = group_names[0]
                bias_result["group_b_name"] = group_names[1]
                
                results[dimension] = bias_result
        
        return results

    def generate_bias_shift_report(
        self,
        pre_attributions: List[List[Tuple[str, float]]],
        post_attributions: List[List[Tuple[str, float]]],
        groups: Optional[Dict[str, Dict[str, List[str]]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive bias shift report across phases.
        
        Args:
            pre_attributions: Attributions from before fine-tuning
            post_attributions: Attributions from after fine-tuning
            groups: Demographic groups to analyze
            
        Returns:
            Comprehensive bias shift report
        """
        if groups is None:
            groups = self.demographic_groups
        
        report = {
            "num_samples": len(pre_attributions),
            "dimensions_analyzed": list(groups.keys()),
            "dimension_reports": {}
        }
        
        for dimension, group_tokens in groups.items():
            if len(group_tokens) >= 2:
                group_names = list(group_tokens.keys())
                group_a_tokens = group_tokens[group_names[0]]
                group_b_tokens = group_tokens[group_names[1]]
                
                amplification = self.calculate_bias_amplification(
                    pre_attributions,
                    post_attributions,
                    group_a_tokens,
                    group_b_tokens
                )
                amplification["group_a_name"] = group_names[0]
                amplification["group_b_name"] = group_names[1]
                
                report["dimension_reports"][dimension] = amplification
        
        # Generate summary
        all_trends = [r.get("bias_trend") for r in report["dimension_reports"].values()]
        report["summary"] = {
            "dimensions_with_amplified_bias": sum(1 for t in all_trends if t == "amplified"),
            "dimensions_with_reduced_bias": sum(1 for t in all_trends if t == "reduced"),
            "dimensions_with_stable_bias": sum(1 for t in all_trends if t == "stable"),
            "overall_trend": max(set(all_trends), key=all_trends.count) if all_trends else "unknown"
        }
        
        return report

    def analyze_bias_patterns(
        self,
        all_attributions: Dict[str, List[Dict[str, Any]]],
        groups: Optional[Dict[str, Dict[str, List[str]]]] = None,
    ) -> Dict[str, Any]:
        """Analyse demographic bias patterns across XAI methods.

        Args:
            all_attributions: ``{method_name: [{"text": ..., "attribution": [(word, score), ...],
                               "prediction": int, "label": int}, ...]}``
            groups: Demographic token groups to test.  Defaults to ``self.demographic_groups``.

        Returns:
            Nested dict ``{method: {dimension: bias_analysis_result}}`` plus a summary.
        """
        if groups is None:
            groups = self.demographic_groups

        results: Dict[str, Any] = {}

        for method_name, attr_dicts in all_attributions.items():
            raw_attributions = [d["attribution"] for d in attr_dicts]

            method_results: Dict[str, Any] = {}
            for dimension, sub_groups in groups.items():
                group_names = list(sub_groups.keys())
                if len(group_names) < 2:
                    continue
                group_a_tokens = sub_groups[group_names[0]]
                group_b_tokens = sub_groups[group_names[1]]
                bias = self.analyze_bias(raw_attributions, group_a_tokens, group_b_tokens)
                bias["group_a_name"] = group_names[0]
                bias["group_b_name"] = group_names[1]
                method_results[dimension] = bias

            results[method_name] = method_results

        results["summary"] = {
            "methods_analyzed": list(all_attributions.keys()),
            "dimensions_analyzed": list(groups.keys()),
            "num_samples_per_method": {m: len(v) for m, v in all_attributions.items()},
        }

        return results
