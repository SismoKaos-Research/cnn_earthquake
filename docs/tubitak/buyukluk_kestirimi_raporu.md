# Kısa Pencereli Dalga Formundan Deprem Büyüklüğü Kestirimi

Katalog çapalı veri kümesi üzerinde üç bölümleme protokolü, fizik tabanına karşı başarım ve hatanın büyüklüğe göre dağılımı

---

## Öz

Kısa bir sismik pencereden deprem büyüklüğünün doğrudan kestirilmesi, erken uyarı zincirinin tespitten sonraki ikinci halkasıdır. Bu çalışmada, spektrogram üzerinde çalışan bir evrişimli ağa iki fiziksel değişkenin (genlik ve uzaklık) eklendiği bir regresyon modeli, 55.568 pencere / 32.868 olay / 183 istasyondan oluşan katalog çapalı bir veri kümesinde değerlendirilmiştir.

Model, fizik tabanını (yerel büyüklük bağıntısının ridge uyarlaması) belirgin biçimde geçmektedir: çifte ayrık bölümlemede ortalama mutlak hata **0,2025**, taban ise 0,3155'tir (oran 0,642). Ancak bu toplam değer yanıltıcıdır. Sınama kümesinin **%71,1'i M≤2,5 olaylardan** oluşmakta ve toplam hata bu bandın hatasıyla belirlenmektedir: M≤2,5 için 0,1430, M>3 için **0,4085** — yaklaşık üç kat.

Daha önemlisi, model belirgin bir **doygunluk** göstermektedir. Yanlılık M≤2 bandında +0,124 iken M3–4 bandında −0,22'ye düşmekte, büyüklük başına **−0,18** eğimle ilerlemektedir; yani küçük depremler olduğundan büyük, büyük depremler olduğundan küçük kestirilmektedir. Bu, erken uyarı bağlamında bilinen bir sorundur (Wang vd., 2023) ve eğitim kümesinin yalnızca %1,2'sinin M>4 olmasıyla tutarlıdır.

Ayrıca M2,5–3,0 bandında model fizik tabanını neredeyse hiç geçememektedir (oran 0,949).

**Anahtar sözcükler:** büyüklük kestirimi, erken uyarı, doygunluk, fizik tabanı, bölümleme protokolü

---

## 1. Giriş

Erken uyarı sistemlerinde büyüklük kestirimi, tespitten sonra gelen ve uyarı kararını doğrudan belirleyen adımdır. Klasik yaklaşım, P dalgasının ilk saniyelerinden türetilen genlik ölçütlerini (τ_p, P_d) uzaklıkla birlikte bir bağıntıya sokmaktır. Derin öğrenme yaklaşımı ise dalga formunu doğrudan girdi olarak almaktadır.

Bu alanda tekrar eden sorun **büyüklük doygunluğudur**: kısa bir pencereden kestirilen büyüklük, büyük depremlerde gerçek değerin altında kalmaktadır. Wang vd. (2023) bu sorunu "erken uyarıda büyük bir güçlük" olarak nitelemekte ve iki hafifletme yolu bildirmektedir — eğitim kümesine daha fazla büyük büyüklüklü kayıt eklemek ve girdi süresini uzatmak. Aynı çalışma, girdi süresi 0,5 saniyeden 3 saniyeye çıkarıldığında doygunluğun belirgin biçimde azaldığını raporlamaktadır.

Bu rapor iki soruyu ayırmaktadır:

1. Model, aynı iki fiziksel değişkenden kurulmuş bir doğrusal tabanı geçmekte midir?
2. Geçiyorsa, bu üstünlük büyüklük ekseninde **nerede** bulunmaktadır?

İkinci soru toplam ortalama mutlak hatadan okunamaz, çünkü sınama kümesi küçük depremlerle ağırlıklıdır. Bu ayrım, projedeki tespit çalışmasında da belirleyici olmuştur: toplulaştırılmış bir ölçütün, alt kümelerde tersine dönebildiği ölçülmüştür.

## 2. Veri

### 2.1 Veri kümesi

`dataset_magreg_catalog_6s`, katalog çapalı 6 saniyelik pencerelerden (`window_post_6s_catalog`) üretilmiştir. Her örnek üç bileşenli bir log-güç spektrogramı ile iki yardımcı sayıldan oluşmaktadır: `log_snr` (istasyonun kendi gürültü tabanına göre genlik) ve `log_distance` (dış merkez uzaklığı).

| | Değer |
|---|---|
| Pencere | 55.568 |
| Olay | 32.868 |
| İstasyon | 183 |
| `distance_km` mevcut | %100 |
| Büyüklük aralığı | 2,0 – 7,7 (ortalama 2,429, ortanca 2,30) |
| Uzaklık ortancası / en büyük | 39,6 km / **56 km** |

Çizelge 1. Veri kümesinin özeti.

### 2.2 Katalog çapalı yeniden yapım

Önceki veri kümesi (`dataset_magclass_dual_6s`) STA/LTA tetikleme kapısından geçmiş pencerelere dayanmakta ve `distance_km` sütunu **%100 boş** olduğundan fizik tabanı hiç hesaplanamamaktaydı. Yeniden yapım 2026-08-18'de tamamlanmıştır.

| | eski (kapılı) | yeni (katalog) |
|---|---|---|
| Pencere / olay / istasyon | 35.836 / 23.228 / 152 | 55.568 / 32.868 / 183 |
| `distance_km` mevcut | %0 | %100 |
| En büyük büyüklük | 6,2 | **7,7** |
| Ortanca `log_snr` | 1,752 | **1,314** |

Çizelge 2. Eski ve yeni veri kümelerinin karşılaştırması.

Tetikleme kapısı **daha büyük depremleri değil, daha gürültülü kayıtları** seçmiştir: büyüklük ortalaması neredeyse değişmemiş (2,457 → 2,429), buna karşılık ortanca `log_snr` 0,44 düşmüştür. Kapı kaldırıldığında hata beklenenin tersine **iyileşmiştir**.

### 2.3 Büyüklük dağılımı ve kapsam sınırı

| Bant | Pencere | Pay |
|---|---|---|
| M ≤ 2,0 | 10.919 | %19,6 |
| M 2,0 – 2,5 | 28.636 | %51,5 |
| M 2,5 – 3,0 | 10.531 | %19,0 |
| M 3,0 – 3,5 | 3.550 | %6,4 |
| M 3,5 – 4,0 | 1.261 | %2,3 |
| M > 4,0 | **671** | **%1,2** |

Çizelge 3. Veri kümesinin büyüklük dağılımı.

İki kapsam sınırı raporun tamamı için bağlayıcıdır. Birincisi, kümenin **%1,2'si M>4'tür**; erken uyarı açısından en önemli bant istatistiksel olarak en zayıf banttır. İkincisi, **en büyük dış merkez uzaklığı 56 km'dir**; dalga formları yalnızca yakın alan için istenmiş olduğundan, uzaklığa bağlı sönümlenme bu kümede ölçülememektedir.

## 3. Yöntem

### 3.1 Mimari

Girdi, üç bileşenin log-güç spektrogramıdır. Evrişimli gövde (`CNNBranch`) bu görüntüyü kodlamakta, çıktısı iki yardımcı sayılla birleştirilmekte ve yoğun bir baş büyüklüğü doğrudan kestirmektedir. Kayıp işlevi L1'dir (doğrudan ortalama mutlak hata üzerinde).

Ham dalga formu üzerinde çalışan LSTM+dikkat kolu mimaride bulunmakta ancak **kapalıdır** (`--channels 2d+aux`); açılması sonucu kötüleştirmektedir (6 s'de 0,182 → 0,189). Bu görev için yinelemeli kol ölü ağırlıktır.

### 3.2 Bölümleme protokolleri

Üç protokol kullanılmıştır ve aralarındaki fark **başarım farkı değil, soru farkıdır**:

- **Çifte ayrık.** Sınama kümesi hem hiç görülmemiş olayları hem de hiç görülmemiş istasyonları içermektedir. İstasyon ezberini dışlar. Üç bağımsız bölümleme ile ölçülmüştür.
- **Tespit ediciyle hizalı.** İstasyon bölümlemesi, birinci aşama tespit edicinin bölümlemesinden **kopyalanmaktadır**. Bu, modelin bir art arda dizilimin (cascade) ikinci aşaması olarak kullanılabilmesi için zorunludur; kopyalanmadığında tespit edicinin sınama istasyonlarının %77'si regresörün eğitim istasyonu olmaktadır.
- **Olay ayrık** (yalnızca tarihsel karşılaştırma için). İstasyon paylaşımına izin verir.

Her protokolde, sınama olaylarından herhangi biri eğitimde de görünüyorsa ilgili satırlar düşürülmektedir.

### 3.3 Karşılaştırma tabanları

İki taban kullanılmaktadır ve her ikisi de **eğitim kümesinde uyarlanıp sınama kümesinde uygulanmaktadır**:

- **Sabit ortalama.** Eğitim ortalamasını her satır için kestirmek.
- **Fizik tabanı.** `ridge(log_snr, log_distance)` — yerel büyüklük bağıntısının doğrusal uyarlaması. Modelin geçmesi gereken taban budur; dalga formundan, genlik ve uzaklığın ötesinde bilgi çıkarıp çıkarmadığını sınar.

## 4. Bulgular

### 4.1 Protokollere göre toplam başarım

| Protokol | OMH | Fizik tabanı | Oran |
|---|---|---|---|
| Çifte ayrık (3 bölümleme) | **0,2023 ± 0,0051** | 0,3116 | 0,657 |
| Tespit ediciyle hizalı (3 tohum) | **0,2329 ± 0,0021** | 0,2940 | 0,792 |

Çizelge 4. Bölümleme protokolüne göre ortalama mutlak hata.

Model her iki protokolde de fizik tabanını geçmektedir. İki gözlem önemlidir.

**Bölümleme değişkenliği tohum değişkenliğinin 2,4 katıdır** (±0,0051'e karşı ±0,0021). Hangi istasyonların sınamaya düştüğü, hangi rastgele tohumun kullanıldığından daha belirleyicidir; tek bölümlemeli bir sonuç bu nedenle yanıltıcıdır.

**Tespit ediciyle hizalı bölümleme daha zor bir sınamadır**, daha kötü bir model değil. 0,2329 değeri çifte ayrık aralığın (0,1977–0,2094) dışındadır. Art arda dizilim sayısı, bağımsız sayıyla model bozulmuş gibi karşılaştırılmamalıdır; bu farklı, daha zor ve **zorunlu** bir sınama kümesidir.

### 4.2 Taban, modelden daha çok hareket etmiştir

Yeniden yapımdan önce yayımlanmış oran 0,620, sonra 0,657'dir — oran **kötüleşmiş**, buna karşılık ortalama mutlak hata 0,218'den 0,2023'e **iyileşmiştir**. Nedeni, `distance_km` her satırda mevcut hâle geldiğinde fizik tabanının güçlenmesidir (0,352 → 0,3116).

Yayımlanmış oranlar, sakatlanmış bir tabanla şişirilmişti. **Hareket eden bir oranın modelden mi tabandan mı geldiği her zaman denetlenmelidir.**

### 4.3 Hata büyüklükle birlikte artmakta ve model doygunlaşmaktadır

Üç çifte ayrık bölümlemenin havuzlanmış sonucu (n = 12.928):

| Bant | n | Pay | OMH | Fizik tabanı | Oran | Yanlılık |
|---|---|---|---|---|---|---|
| M ≤ 2,0 | 2.517 | %19,5 | 0,1241 | 0,2703 | 0,459 | **+0,124** |
| M 2,0 – 2,5 | 6.670 | %51,6 | 0,1501 | 0,2027 | 0,741 | −0,032 |
| M 2,5 – 3,0 | 2.380 | %18,4 | 0,3143 | 0,3313 | **0,949** | −0,209 |
| M 3,0 – 3,5 | 815 | %6,3 | 0,3888 | 0,6650 | 0,585 | −0,225 |
| M 3,5 – 4,0 | 342 | %2,6 | 0,4431 | 1,1340 | 0,391 | −0,219 |
| M > 4,0 | 204 | %1,6 | 0,4291 | 1,6082 | 0,267 | +0,022 |

Çizelge 5. Havuzlanmış hata profili (çifte ayrık, üç bölümleme).

Üç bulgu çıkmaktadır.

**Toplam değer küçük depremlerin değeridir.** Sınama satırlarının %71,1'i M≤2,5'tir. Bu bandın hatası 0,1430, M>3 bandının hatası ise **0,4085**'tir — yaklaşık üç kat. Yayımlanan 0,20 değeri, ağırlıklı olarak M2–2,5 depremlerinde ölçülen bir başarımdır.

**Model doygunlaşmaktadır.** Yanlılık M≤2'de +0,124, M3–4'te −0,22'dir; büyüklük başına eğim **−0,18** magnitüd birimidir. Küçük depremler olduğundan büyük, büyük depremler olduğundan küçük kestirilmektedir. Bu, ortalamaya gerileme davranışının klasik imzasıdır ve Wang vd. (2023) tarafından erken uyarıda merkezî bir güçlük olarak tanımlanan doygunlukla aynı yöndedir. Eğitim kümesinin yalnızca %1,2'sinin M>4 olması bu davranışın beklenen nedenidir.

**M2,5–3,0 bandı bir boşluktur.** Bu bantta model fizik tabanını neredeyse hiç geçmemektedir (oran 0,949; bölümleme başına 1,477 / 0,667 / 1,142). Üç bölümlemenin ikisinde model tabandan **daha kötüdür**. Bu bant, dalga formunun genlik ve uzaklığın ötesinde bilgi taşımadığı bir geçiş bölgesi gibi davranmaktadır.

M>4,0 bandındaki +0,022 yanlılık, bölümlemeler arasında kararsızdır (−0,488 / +0,197 / −0,265) ve yalnızca 204 satıra dayanmaktadır; bu bant için tek bir sayı bildirmek yanıltıcı olur.

### 4.4 Uzaklık ve SGO

| Uzaklık | n | OMH | Fizik tabanı | Oran |
|---|---|---|---|---|
| 0 – 25 km | 682 | 0,2011 | 0,2957 | 0,680 |
| 25 – 50 km | 2.183 | 0,2000 | 0,2654 | 0,754 |
| 50 – 100 km | 481 | 0,1970 | 0,2620 | 0,752 |

Çizelge 6. Uzaklığa göre hata (bölümleme 42). Kümede 56 km'nin ötesinde satır bulunmamaktadır.

Uzaklığın hata üzerinde ölçülebilir bir etkisi görünmemektedir. Bu bir bulgudan çok bir **kapsam sınırıdır**: veri kümesinin tamamı 56 km içindedir, dolayısıyla sönümlenmenin devreye girdiği uzaklık aralığı hiç örneklenmemiştir.

SGO ekseninde hata beklendiği gibi artmaktadır (0,154'ten 0,306'ya), ancak fizik tabanına oran her bantta iyileşmektedir (0,795 → 0,536): model, gürültülü kayıtlarda doğrusal bağıntıya göre **görece** daha iyi durumdadır.

## 5. Tartışma

**Toplam ortalama mutlak hata bu görev için tek başına yeterli bir ölçüt değildir.** 0,2025 değeri doğrudur ve fizik tabanını %36 geçmektedir; ancak sınama kümesi küçük depremlerle ağırlıklı olduğundan bu sayı, erken uyarı kararının duyarlı olduğu bandın başarımını neredeyse hiç yansıtmamaktadır. M>3 için ölçülen 0,4085, uyarı eşiği yakınında **yarım magnitüde yakın** bir belirsizlik anlamına gelmektedir.

**Doygunluk veri kümesi tasarımından kaynaklanmaktadır.** Yanlılığın büyüklükle doğrusal ilerlemesi (−0,18/birim), modelin büyük olayları eğitimde yeterince görmemesiyle tutarlıdır. Wang vd. (2023) tarafından bildirilen iki hafifletme yolu — büyük büyüklüklü kayıt eklemek ve girdi süresini uzatmak — bu kümede doğrudan uygulanabilir niteliktedir; birincisi için M>4 payının %1,2'den yükseltilmesi gerekmektedir.

**Bölümleme protokolü sonucu, tohum seçiminden daha çok belirlemektedir.** Bölümleme değişkenliğinin tohum değişkenliğinin 2,4 katı olması, tek bölümlemeli karşılaştırmaların — özellikle küçük farkların — güvenilmez olduğu anlamına gelmektedir.

**Bir oran, payı kadar paydası da hareket ettiğinde yorumlanamaz.** Yeniden yapımda oranın kötüleşmesi modelin gerilemesinden değil, tabanın güçlenmesinden kaynaklanmıştır. Yayımlanmış oranlar eksik bir uzaklık sütunu nedeniyle şişirilmişti.

### 5.1 Sınırlılıklar

- **Tek bölge, tek derlem.** Dış bir sınama kümesi yoktur.
- **56 km kapsam tavanı.** Uzaklığa bağlı sönümlenme ölçülememektedir; uzak alan başarımı hakkında bu veriden hiçbir şey söylenemez.
- **M>4 için 671 pencere.** Erken uyarı açısından en önemli bant, istatistiksel olarak en zayıf banttır; bu bandın yanlılığı bölümlemeler arasında kararsızdır.
- **Tek mimari ailesi.** Yalnızca bu CNN+yardımcı tasarımı bu titizlikte sınanmıştır.
- **Kalibre edilmiş belirsizlik yoktur.** Model nokta kestirimi üretmektedir; uyarı kararı için gereken güven aralığı bulunmamaktadır.

## 6. Sonuçlar

1. Model fizik tabanını her iki protokolde de geçmektedir: çifte ayrık 0,2023 ± 0,0051 (taban 0,3116), tespit ediciyle hizalı 0,2329 ± 0,0021 (taban 0,2940).
2. Toplam değer sınama kümesinin büyüklük dağılımı tarafından belirlenmektedir: satırların %71,1'i M≤2,5'tir ve bu bandın hatası (0,1430), M>3 bandının hatasının (0,4085) üçte biridir.
3. Model doygunlaşmaktadır: yanlılık büyüklük başına −0,18 magnitüd birimi eğimle ilerlemektedir. Eğitim kümesinin %1,2'sinin M>4 olması beklenen nedendir.
4. M2,5–3,0 bandında model fizik tabanını geçmemektedir (oran 0,949; üç bölümlemenin ikisinde tabandan kötü).
5. Bölümleme değişkenliği tohum değişkenliğinin 2,4 katıdır; tek bölümlemeli sonuçlar bildirilmemelidir.
6. Yeniden yapımda oranın kötüleşmesi tabanın güçlenmesinden kaynaklanmıştır; hareket eden oranlarda pay ve payda ayrı ayrı denetlenmelidir.

Bu rapor, modelin büyüklük kestiriminde başarısız olduğunu göstermemektedir. Gösterdiği şey, **yayımlanan toplam hatanın erken uyarı için önemli olan bandı temsil etmediği** ve mevcut hâliyle modelin büyük depremleri sistematik olarak küçük kestirdiğidir.

## Kaynaklar

Wang, Y., Li, X., Wang, Z., & Liu, J. (2023). Deep learning for magnitude prediction in earthquake early warning. *Gondwana Research*, 123, 164–173.

---

*Ölçümler `scripts/magnitude_error_profile.py` ile yapılmıştır; eğitim kayıtları `logs/magreg_*.log` altındadır. Toplam değerler eğitim kayıtlarıyla dört ondalık basamağa kadar örtüşmektedir.*
