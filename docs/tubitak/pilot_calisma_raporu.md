# Pilot Çalışma Raporu: Kademeli Deprem Tespit ve Büyüklük Kestirim Sistemi

**Taslak — 06.09.2026.** Bu belge, projede bugüne kadar yapılmış ölçümlerin
bütünleşik bir özetidir. Her sayının yanında hangi protokolle ve hangi taban
karşısında elde edildiği belirtilmiştir. Sınırlamalar Bölüm 7'de toplu olarak
verilmiştir; bazı sonuçlar tek istasyona dayanmakta olup genellenemez.

---

## 1. Özet

Bu çalışma, sürekli sismik kayıt üzerinde çalışan iki kademeli bir sistemi
ölçmektedir: birinci kademe kısa pencereli bir derin öğrenme **tespit
edicisi**, ikinci kademe aynı pencereden **büyüklük kestirimi** yapan bir
bağlanım modelidir. Beş başlıkta bulgu sunulmaktadır.

**(1) Sınama kümesi başarımı sürekli veriye aktarılamamaktadır.** Sınama
kümesinde 0,9896 AUC elde eden 6 saniyelik tespit edici, sürekli gürültü
üzerinde 0,80 ortanca puan vermekte ve 0,5 eşiğinde günde **12.599** alarm
üretmektedir — sınama kümesinden çıkarsanan tahmin 257 idi.

**(2) Olay düzeyinde AUC ile işletme çalışma noktası tespit edicileri ters
sırada dizmektedir.** 1978 tarihli klasik STA/LTA yöntemi her SGO kesiminde
AUC bakımından üstün, buna karşılık bir dağıtımın seçeceği her alarm bütçesinde
geridedir.

**(3) Büyüklük kestirimi fizik tabanını geçmektedir**, ancak toplam hata küçük
depremlerin hatasıdır: M ≤ 2,5 için 0,1430, M > 3 için **0,4085**.

**(4) İki istasyonun uyuşması şart koşulduğunda kazanç istasyon uzaklığına
bağlıdır.** 63 km aralıklı çiftte yanlış alarmlar bağımsızlık varsayımının
öngördüğünün 29–58 katı örtüşmekte, 144 km'de varsayım geçerli olmaktadır.

**(4b) Büyüklük kestirimini katalog yerine gerçek tespit alarmı üzerine
çapalamak, göreli başarımdan pratikte hiçbir şey götürmemektedir.** Ham fark
+0,0071 OMH'dir, ancak alarm çapalı kurumun bütün tabanları daha yüksektir;
her kurum kendi tabanına oranlandığında fark −0,004'e inmektedir.

**(5) Kademeli sistem, tespit edilen olayların %62,7'sinde** (6 s büyüklük
penceresi ile) hem tespiti hem büyüklük kestirimini S dalgası varmadan
tamamlamaktadır. Belirleyici kısıt tespit edici değil, **büyüklük penceresinin
uzunluğudur**.

---

## 2. Veri ve yöntem

### 2.1 Kaynaklar

Dalga formu kaynağı **ağ koduna göre** seçilmektedir; olay/sürekli ayrımına göre
değil. KO (Kandilli) ağı FDSN üzerinden doğrudan alınmakta (24 saat ~13 s), TU
(AFAD) ağı ise yalnızca TDVMS e-posta kuyruğundan gelmektedir (istasyon-gün
başına ~2 dakika). TU'nun tek üstünlüğü **istasyon ayrıklığıdır**: FDSN'in 163
KO istasyonunun 156'sı zaten eğitim kümesindedir.

06.09.2026'da ölçülmüştür: **bir TDVMS isteği tek istasyon döndürmektedir.**
İstek gövdesi bir istasyon dizisi kabul etmekte ve yanıt "talep ettiğiniz
istasyon/istasyonlara ait veri" demekte, ancak arşivde tek istasyon
bulunmaktadır. N istasyon N kuyruk yuvasına mal olmaktadır.

### 2.2 Katalog

Depodaki üç yerel katalog da, KOERI olarak adlandırılmalarına karşın **AFAD
verisidir**; EventID'lerin tamamı AFAD kimlikleridir. Kullanımda olan dosya
bölgedeki 5.770 olayın 1.688'ini kaçırmaktaydı (253'ü M ≥ 4,0) — pratikte
Şubat 2025 Santorini–Amorgos sürüsünün tamamı. Katalog 30.08.2026'da yeniden
kurulmuştur (413.785 olay, M ≥ 1,5, 2000–2026). Etiketlerin yeniden türetilmesi
bir taban oranı %25,1'den %39,9'a taşımıştır. **Dalga formları gerçekten
KOERI'ye aittir**; yanlış olan yalnızca katalog atfıydı.

### 2.3 Taban ilkesi

Bu projede hiçbir model sabit bir kestirici karşısında değerlendirilmemektedir.
Her görev için **koşullu bir taban** kullanılmaktadır: tespit için doğru
parametrelenmiş STA/LTA, büyüklük için `ridge(log_snr, log_distance)` — yerel
büyüklük bağıntısının uyarlanmış hâli. Sabit taban iki kez yanlış pozitif
üretmiştir.

Karşılaştırma `(AUC − taban) / (1 − taban)` üzerinden yapılmalıdır; ham AUC
üzerinden değil, çünkü farklı pencereler farklı taban bırakmaktadır.

---

## 3. Birinci kademe: tespit

### 3.1 Sınama kümesi

**Çizelge 1.** Küratörlü sınama kümesi (dengeli sınıflar, varış çapalı
pozitifler, madenlenmiş negatifler).

| Model | AUC | Taban | Yakalanan boşluk |
|---|---|---|---|
| 6 s, 1B kol (`cnn-lstm`) | **0,9896** | 0,9049 | %89,1 |
| 6 s, 2B kol | 0,9882 | 0,9049 | %87,6 |
| 3 s, çapalı | 0,9805 | 0,8481 | %87,2 |
| P-dalgası kolu, 3,4 s | 0,8712 | 0,6679 | %61,2 |

**Pencereyi yarıya indirmek modeli zayıflatmamakta, görevi zorlaştırmaktadır.**
Taban 0,9049'dan 0,8481'e düşmekte (3 s pencere varış sonrası daha az enerji
yakalamaktadır), buna karşılık model her iki durumda da mevcut boşluğun aynı
oranını yakalamaktadır.

### 3.2 Sürekli veri: eşik aktarılamamaktadır

MANT istasyonunun **728 günlük** kesintisiz kaydı, 10.487.211 pencere olarak
puanlanmıştır. Pencereler ayrıktır (adım = pencere boyu), dolayısıyla alarm
sayısı aynı zamanda **bağımsız karar sayısıdır**.

**Çizelge 2.** Arka plan dağılımı ve 0,5 eşiğindeki alarm sayısı.

| Kol / istasyon | Arka plan ortancası | 0,5'te alarm/gün |
|---|---|---|
| 6 s, MANT | 0,8019 | **12.599** |
| P-dalgası (doğal), MANT | 0,3037 | 45 |
| 6 s, DEMI | 0,8358 | — |
| 6 s, GCAM | 0,1205 | — |

6 saniyelik model, sınama kümesinde ayarlanmış 0,5 eşiğinde **sessiz bir
istasyon gününün %92,7'sini** işaretlemektedir.

**Nedeni ölçülmüştür: bir sessizlik bölgesi.** Genlik madenciliği negatif
sınıfa, fizik ise pozitif sınıfa bir taban koymakta; aradaki genlik bandında
modelin hiçbir sınıftan eğitim verisi bulunmamaktadır. Sürekli arka plan tam
olarak orada, 0,11 σ düzeyindedir. Gerçek eğitim gürültüsü ölçeklendirilerek
ölçülmüştür: alarm oranı eğitim ortancasında %1,9 iken, onda birinde %86,
yüzde birinde %100'dür. Her iki P-dalgası modeli tekdüzedir, dolayısıyla bu
madenciliğin kendisinden kaynaklanmamaktadır.

**Yanlış alarm oranı istasyona özgüdür.** Aynı model ve aynı ön işleme, arka
plan ortancasını MANT'ta 0,8019, DEMI'de 0,8358, GCAM'de ise 0,1205 vermektedir;
sıralama istasyonun uzun dönemli σ değerine göredir.

### 3.3 AUC ile çalışma noktası ters sırada dizmektedir

**Çizelge 3.** SGO ≥ 3 koşulunda 13.056 olay, 728 gün.

| Tespit edici | Olay AUC | Duyarlılık @100/gün | @10/gün | @1/gün |
|---|---|---|---|---|
| STA/LTA (klasik) | **0,9795** | **0,928** | 0,548 | 0,132 |
| P-dalgası (madenlenmiş) | 0,9622 | 0,903 | 0,627 | 0,183 |
| P-dalgası (doğal) | 0,9516 | 0,861 | 0,573 | 0,219 |
| 6 s | 0,9403 | 0,864 | **0,741** | **0,316** |

AUC sütunu ile 10/gün sütunu **tam olarak ters sıradadır.** 1978 tarihli bir
STA/LTA her SGO kesiminde AUC bakımından kazanmakta, bir dağıtımın seçeceği her
alarm bütçesinde kaybetmektedir; 1 alarm/gün bütçesinde 6 saniyelik model
**2,4 kat** fazla olay bulmaktadır.

AUC tüm ROC eğrisini bütünlemektedir; işletme çalışma noktası ise eğrinin tek
bir uç köşesinde, burada YPO 7,4×10⁻⁴ düzeyinde bulunmaktadır. **Yalnızca AUC
üzerinden bildirilen her tespit edici karşılaştırması, dağıtım koşullarında
tersine dönebilir.**

### 3.4 Açıklanamayan alarmlar günlük döngü göstermektedir

10 alarm/gün bütçesinde açıklanamayan alarmların gündüz/gece oranı 1,73–2,07
kat, tepe noktası 12:00–15:00'tir. Bu, önemli bir bölümünün kataloglanmamış
deprem değil **kültürel gürültü** olduğunu göstermekte ve yanlış alarm üst
sınırını daraltmaktadır.

---

## 4. İkinci kademe: büyüklük kestirimi

### 4.1 Katalog çapalı kurum (çok istasyon)

**Çizelge 4.** `dataset_magreg_catalog_6s`, ortalama mutlak hata.

| Protokol | OMH | Taban | Oran |
|---|---|---|---|
| Çifte ayrık (3 bölümleme) | **0,2023 ± 0,0051** | 0,3116 | 0,657 |
| Tespit ediciyle hizalı (3 tohum) | 0,2329 ± 0,0021 | 0,2940 | 0,792 |

**Tespit ediciyle hizalı bölümleme daha zor bir sınamadır, daha kötü bir model
değil.** 0,2329 değeri çifte ayrık aralığın (0,1977–0,2094) dışındadır ve
kademeli sistem için **zorunlu** olan sınama kümesidir: hizalama yapılmadığında
tespit edicinin sınama istasyonlarının %77'si bağlanım modelinin **eğitim**
istasyonu olmaktadır.

### 4.2 FDSN kurumu, dalga formu ile sınırlı (çok istasyon)

06.09.2026'da ölçülmüştür. `--channels 1d+2d`: yardımcı vektör kapalıdır, çünkü
`log_distance` bir katalog hiposantrı gerektirmekte, taze bir tespitte ise böyle
bir bilgi bulunmamaktadır.

| | OMH |
|---|---|
| Model (`1d+2d`), 3 bölümleme | **0,4203 ± 0,0165** |
| `ridge(log_snr)` — bilgi eşleştirilmiş taban | 0,6054 ± 0,0508 |
| `ridge(log_snr, log_distance)` | 0,5481 ± 0,0583 |
| Sabit kestirici | 0,7233 ± 0,0595 |

**Model, uzaklık bilgisine sahip tabanı üç bölümlemenin üçünde de
geçmektedir** — kendisi uzaklığı görmediği hâlde.

**Taban modelden daha oynaktır.** Modelin bölümlemeler arası yayılımı 0,0165,
tabanınki 0,0583'tür — **3,5 katı**. Dolayısıyla tabana oran 0,83'ten 0,68'e
hareket etmekte, ancak bunun nedeni modelin değil tabanın değişmesidir. Oranın
tek başına bildirilmesi bunu modelin değişimi gibi gösterirdi.

**Yardımcı vektörün bedeli ölçülmüştür.** Aynı 578 sınama satırında, aynı
tohumla: `2d+aux` 0,3572, `1d+2d` 0,4094. Uzaklığın esirgenmesi yaklaşık
**0,05 OMH**'ye mal olmaktadır (tek bölümleme).

### 4.3 Toplam değer küçük depremlerin değeridir

6 saniyelik kurumda sınama satırlarının **%71,1'i M ≤ 2,5**'tir ve bu bandın
hatası 0,1430, M > 3 bandının hatası ise **0,4085**'tir — yaklaşık üç kat.
Yayımlanan 0,20 değeri, ağırlıklı olarak M 2–2,5 depremlerinde ölçülmüş bir
başarımdır. Uyarı eşiği yakınında belirsizlik **yarım magnitüde yakındır**.

**Model doygunlaşmaktadır.** Yanlılık M ≤ 2 bandında +0,124 iken M 3–4 bandında
−0,22'ye inmekte, büyüklük başına **−0,18** magnitüd birimi eğimle
ilerlemektedir: küçük depremler olduğundan büyük, büyük depremler olduğundan
küçük kestirilmektedir. Eğitim kümesinin yalnızca **%1,2'sinin M > 4** olması
beklenen nedendir. Wang vd. (2023) iki hafifletme yolu bildirmektedir: büyük
büyüklüklü kayıt eklemek ve girdi süresini uzatmak.

---

## 5. Kademeli sistem

### 5.1 Alarm çapalı kurum

Kademeli sistemin ikinci kademesi, kataloğun değil **birinci kademenin**
ürettiği pencereleri görmelidir. `dataset_magreg_alarm_10s` tam olarak budur:
pencereler tespit edicinin alarm zamanlarından kesilmiştir.

**Çizelge 5.** MANT, 10 s pencere, `--channels 1d+2d`, tek tohum, n = 1.150.

| | OMH |
|---|---|
| Model (alarm çapalı, dalga formu ile sınırlı) | **0,1821** |
| `ridge(log_snr)` — bilgi eşleştirilmiş taban | 0,3338 |
| `ridge(log_snr, log_distance)` | 0,2403 |
| Sabit kestirici | 0,5626 |

RMSE 0,2936, R² 0,8355.

Model, **uzaklığı görmediği hâlde uzaklık bilgisine sahip tabanı da**
geçmektedir (0,1821'e karşı 0,2403). Bu, işletme açısından anlamlı olan
karşılaştırmadır: alarm anında hiçbir bileşenin hiposantra erişimi
bulunmamaktadır.

### 5.2 Kademeli sistemin bedeli

Katalog çapalı denetim koşusu (`dataset_magreg_cont_10s`, aynı istasyon, aynı
pencere boyu, aynı yapılandırma) tamamlanmıştır.

**Çizelge 6.** Alarm çapalı ile katalog çapalı kurumun karşılaştırılması.

| | OMH | `ridge(log_snr)` | `ridge(log_snr, log_distance)` | Sabit | n |
|---|---|---|---|---|---|
| Katalog çapalı | 0,1750 | 0,3184 | 0,2072 | 0,5426 | 1.953 |
| Alarm çapalı | 0,1821 | 0,3338 | 0,2403 | 0,5626 | 1.150 |

Ham fark **+0,0071 OMH (%+4,1)**'dir. Ancak bu iki kurum eşit zorlukta
değildir: **alarm çapalı kurumun bütün tabanları daha yüksektir**, yani kurum
daha zordur. Her kurum kendi tabanına oranlandığında:

| Taban | Alarm çapalı | Katalog çapalı | Fark |
|---|---|---|---|
| `ridge(log_snr)` (bilgi eşleştirilmiş) | 0,546 | 0,550 | **−0,004** |
| `ridge(log_snr, log_distance)` | 0,758 | 0,845 | −0,087 |
| Sabit kestirici | 0,324 | 0,323 | +0,001 |

**Kademeli sistemin bedeli, kurumun zorluğuyla açıklanmaktadır.** Model, alarm
çapalı kurumda tam olarak kurumun zorlaştığı kadar kötüleşmektedir; kendi
tabanına göre ölçüldüğünde beceri katkısı denetim koşusundan **ayırt
edilememektedir** (0,546'ya karşı 0,550).

İşletme açısından sonuç şudur: **büyüklük kestirimini katalog varışı yerine
gerçek tespit alarmı üzerine çapalamak, göreli başarımdan pratikte hiçbir şey
götürmemektedir.** Bu, kademeli sistemin dağıtılabilirliği açısından temel
bulgudur, çünkü alarm anında katalog bulunmamaktadır.

Bu, projenin yinelenen dersinin bir örneğidir: **oran değiştiğinde önce tabanın
mı yoksa modelin mi hareket ettiği sorulmalıdır.** Burada hareket eden tabandır.

**Uyarı.** İki koşu farklı sınama kümeleri üzerindedir (1.150'ye karşı 1.953) ve
eşleştirilmiş bir karşılaştırma değildir; alarm çapalı kurum yalnızca tespit
edilmiş olayları içermektedir. Her iki koşu tek tohumludur ve tek istasyona
dayanmaktadır.

### 5.3 Karşılaştırılabilirlik uyarısı

0,1821 ile 0,4203 doğrudan karşılaştırılamaz. Üç fark bulunmaktadır:

| | Alarm çapalı (MANT) | FDSN |
|---|---|---|
| İstasyon | **1** | 161 |
| Ortanca büyüklük | 1,90 | 3,50 |
| M < 2,5 payı | **%77,8** | %17,0 |
| Sabit kestirici tabanı | 0,5517 | 0,8070 |
| Bölümleme | olay ayrık (tek istasyon zorunlu kılmaktadır) | çifte ayrık |

Sabit kestirici tabanına oranlandığında fark 2,3 kattan yaklaşık 1,8 kata
inmektedir. **Toplam OMH büyük ölçüde kurumun büyüklük dağılımının bir
okumasıdır**; daha çok küçük deprem içeren bir kurum, daha iyi bir model gibi
görünmektedir.

---

## 6. İki istasyonla uyuşma

**Çizelge 7.** Ölçülen uyuşma oranı ile bağımsızlık varsayımının öngördüğü oran.

| Çift | Uzaklık | Ölçülen 2/2 (gün) | Bağımsız olsaydı | Fazlalık | Duyarlılık |
|---|---|---|---|---|---|
| MANT–DEMI, P-dalgası | 63 km | 0,425 | 0,0148 | **28,8×** | 0,369 |
| MANT–DEMI, 6 s | 63 km | 0,709 | 0,0123 | **57,5×** | 0,563 |
| MANT–GCAM, P-dalgası | 144 km | 0,044 | 0,0408 | **1,1×** | 0,371 |
| MANT–GCAM, 6 s | 144 km | 0,069 | 0,0318 | **2,2×** | 0,444 |

**İkinci istasyonun getirisi uzaklığa bağlıdır.** 63 km aralıklı çiftte iki
istasyonun yanlış alarmları bağımsızlık varsayımının öngördüğünün 29–58 katı
örtüşmektedir; iki istasyon hava koşullarını, bölgesel gürültü alanını ve
kültürel gürültünün günlük döngüsünü paylaşmaktadır. 144 km'de varsayım
geçerlidir (1,1–2,2 kat).

**Uzak çift, aynı duyarlılıkta on kat daha az yanlış bildirim üretmektedir:**
23 günde bir, 2,4 güne karşılık. Bulgu her iki tespit kolunda bağımsız olarak
elde edildiğinden tek bir modelin özelliği değildir.

**Uyarı.** 144 km'deki hücreler 7 ve 11 bildirime dayanmaktadır. Bu sayılarla
1 kat ile 3 kat ayırt edilememektedir; dolayısıyla "144 km'de tam bağımsızlık"
**iddia edilememektedir**, yalnızca fazlalığın 63 km'dekinden çok daha küçük
olduğu söylenebilmektedir.

---

## 7. Zamanlama: kademeli sistem S dalgasını geçiyor mu?

Bir erken uyarı sisteminin üretmesi gereken çıktı yalnızca "deprem var" bilgisi
değil, büyüklük kestirimiyle birliktedir. `alarm_epoch` tespit penceresinin
**sonudur**; ikinci kademe kendi penceresi dolmadan çıktı verememektedir.
Kademeli sistem S dalgasını yalnızca `dt_vs_s + (W − P − L) < 0` koşulunda
geçmektedir.

**Çizelge 8.** 11.188 tespit edilen olay; tespit tek başına %69,5'inde S'den
öncedir.

| Büyüklük penceresi | Eklenen gecikme | Kademeli sistem S'den önce | Başabaş uzaklık |
|---|---|---|---|
| 6 s | +4,0 s | **%62,7** | 50 km |
| 10 s | +8,0 s | %45,0 | 100 km |
| 20 s | +18,0 s | %4,6 | 200 km |

**Çizelge 9.** Uzaklığa göre (tespit edilen olaylar içindeki oran).

| Uzaklık (km) | Olay | Yalnız tespit | W = 6 s | W = 10 s | W = 20 s |
|---|---|---|---|---|---|
| 0 – 25 | 125 | %41,6 | %0,8 | %0,0 | %0,0 |
| 25 – 50 | 252 | %66,7 | %6,7 | %0,4 | %0,0 |
| 50 – 100 | 9.353 | %68,5 | **%62,9** | %42,1 | %0,6 |
| 100 – 150 | 654 | %81,0 | %79,2 | %77,7 | %0,9 |
| 150 – 200 | 245 | %72,2 | %70,6 | %69,8 | %27,8 |
| 200 – 300 | 308 | %72,4 | %69,2 | %67,5 | %61,4 |
| 300 – 500 | 251 | %87,3 | %83,7 | %82,1 | %75,7 |

Ölçülen S−P eğimi 100 km başına 10,77 s'dir (eşdeğer 9,28 km/s); bir Vp değeri
varsayılmamış, veriden uyarlanmıştır.

**Belirleyici kısıt büyüklük penceresidir.** Tespit tek başına %69,5'e
ulaşmaktadır; 6 saniyelik pencere ~7 puan, 10 saniyelik pencere 24,5 puan,
20 saniyelik pencere 64,9 puan kayba yol açmaktadır.

**Kör bölge ölçülmüştür.** 0–25 km bandında kademeli sistem olayların yalnızca
%0,8'inde yetişmektedir. Bu, erken uyarı yazınında bilinen olgudur.

**100–150 km bandında pencere uzunluğu neredeyse bedelsizdir** (%79,2'ye karşı
%77,7). Daha uzun pencerenin büyüklük doğruluğu kazancı bu bantta bedelsiz elde
edilmektedir.

Bölüm 4 ile birlikte okunduğunda bu, **ölçülmüş bir ödünleşim eğrisidir**:
pencereyi uzatmak kestirimi iyileştirmekte, erken uyarı yeteneğini ise
azaltmaktadır. Tek bir "en iyi" pencere uzunluğu bulunmamakta, seçim uygulama
amacına bağlı olmaktadır.

---

## 8. Sınırlamalar

**Yanlış alarm sayıları üst sınırdır.** Kataloğa uymayan bir alarm ya yanlış
pozitiftir ya da AFAD'ın kataloglamadığı bir depremdir; bu çalışma ikisini
ayırt edememektedir. Günlük döngü bulgusu üst sınırı daraltmaktadır.

**Duyarlılık yalnızca istasyonun kaydettiği olaylarda sorulmuştur.** MANT'ta
kataloglanmış 47.522 olayın yalnızca **%27,5'i** SGO 3 eşiğine ulaşmaktadır
(ortanca 1,39). Tüm olaylara karşı puanlandığında olay AUC'si 0,67–0,73'tür;
bu sayı kataloğun erişimini tanımlamakta, modeli değil.

**Doygunlaşmış model sahte duyarlılık göstermektedir.** 6 saniyelik kol, %95
arka plan oranında "%97,8 duyarlılık, P'den 7 s önce" bildirmişti; bu aritmetik
sonuçtur, tespit değil. Her duyarlılık değerinin yanında arka plan oranı
verilmelidir.

**Alarm zamanları nicemlenmiştir.** 6 saniyelik kolun adımı 6 s olduğundan
`dt_vs_s` ±6 s çözünürlüktedir; **0–50 km bandı bu çözünürlüğün altındadır** ve
bu bantların sayıları yön göstericidir.

**Tek istasyonlu kurumlar genellenemez.** `magreg_alarm_10s` ve
`magreg_cont_*` yalnızca TU.MANT içermektedir; bu kurumlardan elde edilen her
sayı **o istasyonun** kestiricisidir. İstasyon aktarımı iddiası yalnızca FDSN
kurumundan yapılabilmektedir.

**Yakın marj karşılaştırmaları tek tohuma dayanmaktadır.** Depo kayıtlarında
üç tohumla yinelenen iki karşılaştırmadan biri **işaret değiştirmiştir**;
yaklaşık 0,01–0,02'nin altındaki farklar yerleşik etki sayılmamalıdır.

**Zamanlama doğruluk değildir.** Bölüm 7 kestirimin *ne zaman hazır olduğunu*
hesaplamaktadır, *ne kadar doğru olduğunu* değil.

---

## 9. Sonuç

1. Kısa pencereli tespit edici, küratörlü sınamada tabanın üzerindeki boşluğun
   %89'unu yakalamaktadır; ancak bu başarım **sürekli veriye doğrudan
   aktarılamamaktadır** ve çalışma noktası ölçülen arka plandan türetilmelidir.
2. **Yalnızca AUC ile yapılan tespit edici karşılaştırmaları dağıtım
   koşullarında tersine dönebilmektedir**; bu, klasik bir yöntemin somut
   örneğiyle gösterilmiştir.
3. Büyüklük kestirimi fizik tabanını her protokolde geçmektedir, ancak toplam
   değer küçük depremlerin değeridir ve model doygunlaşmaktadır.
4. Kademeli sistem MANT'ta **0,1821 OMH** ile uzaklık bilgisine sahip tabanı da
   geçmektedir; bu tek istasyonluk bir sonuçtur. **Alarm üzerine çapalamanın
   bedeli, kendi tabanına oranlandığında −0,004'tür** — yani pratikte yoktur.
5. İkinci istasyonun getirisi **uzaklığa bağlıdır**; ~100 km'nin altındaki
   çiftler için bağımsızlık varsayılmamalıdır.
6. Kademeli sistem, 6 saniyelik büyüklük penceresiyle tespit edilen olayların
   **%62,7'sinde** S dalgasından önce sonuçlanmaktadır; belirleyici kısıt
   büyüklük penceresidir.

**Önerilen sonraki adımlar.** (i) 0–50 km bandının sık pencereleme ile yeniden
taranması. (ii) Aynı zamanlama çözümlemesinin P-dalgası kolu ve klasik yöntem
için yinelenmesi. (iii) Alarm çapalı kurumun DEMI ve GCAM istasyonlarında da
üretilerek kademeli sonucun istasyon aktarımı iddiasına açılması. (iv) Büyüklük
hatasının, S'den önce yetişen olaylar alt kümesinde ayrıca raporlanması.

---

*Ölçümler `sk` komut satırı üzerinden yapılmıştır; her koşu `runs/` altında
argv, git işlemesi, çalışma ağacının temiz olup olmadığı, veri kümesi kimliği,
tohumlar ve ölçütlerle birlikte kayıtlıdır. Ayrıntılı raporlar:
`surekli_veri_yanlis_alarm_raporu.md`, `buyukluk_kestirimi_raporu.md`,
`pilot_kademeli_zaman_raporu.md`.*
