# Katalog Kusuru Teknik Raporu

## deprem_katalog_utc.csv neden kullanılamaz durumdadır

**Tarih:** 01.09.2026 · **Tespit:** 30.08.2026 · **Durum:** giderildi

---

## 1. Özet

Projenin tahmin (forecasting) etiketleri, `Sismokaos/data_downloader/catalogs/`
dizinindeki yerel bir olay kataloğundan üretilmektedir. Bu raporun konusu olan
`deprem_katalog_utc.csv` dosyası, projenin başka bir bölümünden devralınmış olup
**üç ayrı ve birbirinden bağımsız nedenle kullanılamaz durumdadır**:

1. **Zaman ekseni eksiktir** — dosya 01.01.2010'da başlamakta, öncesinde **sıfır**
   kayıt bulunmaktadır. Güncel katalog ise 04.01.2000'e kadar gitmektedir.
2. **İçerik eksiktir** — kapsadığı dönemde bile bölgeye ait olayların **%38,8'ini**
   içermemektedir; eksiklerin **278'i M≥4,0**, en büyüğü **M6,0**'dır.
3. **Eksiklik düzensizdir** — çeyrek bazında eksik oranı %6,7 ile %89,9 arasında
   değişmektedir; dolayısıyla sabit bir eşik veya basit bir kesme ile
   açıklanamaz ve düzeltme katsayısıyla telafi edilemez.

Dosya **AFAD verisidir** — depoda "KRDAE/KOERI kataloğu" olarak anılması hatalıdır.
Söz konusu pencerede tüm EventID değerleri AFAD eventID'leridir ve büyüklük ile
koordinatlar AFAD API'siyle basılan basamağa kadar aynıdır. Sorun kaynağın kimliği
değil, **kopyanın eksikliğidir**.

---

## 2. Kusurun ölçümü

Karşılaştırma AFAD'ın kamuya açık API'si esas alınarak yapılmıştır. Ölçüt hücresi
proje boyunca kullanılan hücredir: BODT istasyonundan 400 km yarıçap, M≥2,5.

### 2.1 Zaman kapsamı

| | başlangıç | bitiş | satır |
|---|---|---|---|
| `deprem_katalog_utc.csv` | 01.01.2010 | 27.06.2026 | 482.898 |
| güncel katalog | 04.01.2000 | 29.08.2026 | 576.829 |

2010 öncesinde dosyada **sıfır** kayıt bulunmaktadır. Bu, tek başına belirleyici
bir kusurdur: `catalog_forecast_report.md`'de belgelenen bölge bazlı tahmin
çalıştırması `--catalog-span 2000-01-01 2026-08-12` parametresiyle koşmaktadır.
Bu dosya ile çalıştırıldığında modele **on yıllık boş bir aralık** verilmekte,
1. katlamanın AUC değeri **0,3098**'e — yani rastlantının belirgin biçimde
altına — düşmektedir.

### 2.2 İçerik eksikliği

Yalnızca dosyanın **kapsadığı** dönem (01.05.2024 – 27.06.2026) karşılaştırılmıştır;
2010 öncesi boşluk bu tabloya dâhil değildir.

| | olay sayısı |
|---|---|
| güncel katalog | 5.634 |
| `deprem_katalog_utc.csv` | 3.449 |
| **eksik** | **2.185 (%38,8)** |

Eksiklerin **278'i M≥4,0** büyüklüğündedir; kaçırılan en büyük olay **M6,0**'dır.
Bu, dosyanın küçük olayları elemiş olabileceği yönündeki açıklamayı geçersiz
kılmaktadır: M6,0 bir deprem hiçbir tamamlılık eşiğinin altında kalmaz.

### 2.3 Eksikliğin dağılımı

| çeyrek | eksik / toplam | eksik oranı |
|---|---|---|
| 2024Ç2 | 70 / 155 | %45,2 |
| 2024Ç3 | 87 / 196 | %44,4 |
| 2024Ç4 | 71 / 197 | %36,0 |
| **2025Ç1** | **1.320 / 1.468** | **%89,9** |
| 2025Ç2 | 156 / 424 | %36,8 |
| 2025Ç3 | 86 / 1.283 | %6,7 |
| 2025Ç4 | 83 / 1.127 | %7,4 |
| 2026Ç1 | 125 / 463 | %27,0 |
| 2026Ç2 | 187 / 321 | %58,3 |

Oranın %6,7 ile %89,9 arasında salınması kusurun niteliğini ortaya koymaktadır.
Sabit bir büyüklük eşiği ya da tek seferlik bir kesme, çeyrekten çeyreğe on üç
kat değişen bir eksiklik üretemez.

### 2.4 Eksikliğin odağı

2025Ç1'deki 1.320 eksik olayın **1.204'ü Şubat 2025'e**, bunların da **1.201'i
"Ege Denizi" konumuna** aittir. Bu, **Santorini–Amorgos deprem dizisidir** —
kayıt penceresindeki en büyük sismik olaydır ve dosyada fiilen bulunmamaktadır.

Eksiklik **zamansal değil uzamsaldır**. Dosya Şubat 2025 için ülke genelinde
1.930 satır içermekte, ancak bunların yalnızca 153'ü "Ege Denizi"dir; AFAD ise tek
başına bizim ölçüt hücremizde 1.225 kayıt vermektedir. Dolayısıyla özgün indirme
işlemi kesintiye uğramamış, **kıyı ötesi olayları yetersiz örneklemiştir**.

---

## 3. Kusurun neden fark edilmediği

Dosya, hatalı veriye ilişkin olağan uyarı işaretlerinin hiçbirini vermemektedir.
Boş değildir (482.898 satır), bozuk değildir, biçimi doğrudur ve **27.06.2026'ya
kadar günceldir**. Satır sayısına, dosya boyutuna veya son kayıt tarihine bakan
herhangi bir denetimden geçer.

Kusur ancak dosya **kendi kaynağıyla** karşılaştırıldığında görünür hâle
gelmektedir. Bu karşılaştırma yapıldığında AFAD API'si aynı hücre için 1,67 kat
daha fazla olay vermiş; olay bazında (EventID, büyüklük, koordinat) eşleştirme
ise elde tutulan kayıtların **%100'ünün** birebir tuttuğunu göstermiştir. Bu
sonuç kritiktir: dosya **farklı bir katalog değil, eksik bir alt kümedir**.

---

## 4. Etkilenen çalışmalar

`deprem_katalog_utc.csv` dosyasını okuyan bileşenler:

- `src/sismokaos/forecasting/label_sweep.py` (varsayılan)
- `src/sismokaos/forecasting/cnn_lstm_forecast.py`
- Kaotik öznitelik çalışmasının dört betiği (`chaos_univariate_screen`,
  `chaos_forecast`, `chaos_config_sweep`, `chaos_station_replication`)

Kaotik öznitelik çalışmasının etiketleri bu dosyadan **birebir yeniden
üretilmiştir**: 233 olay → 4.343 saatte 1.092 pozitif (%25,1), ki bu değer
`logs/chaos_screen.log` kaydıyla tam olarak örtüşmektedir. Aynı pencerede
`data_large.csv` 339 olay → 1.492 pozitif (%34,3) vermektedir. **Sonuç olarak
tahmin çalışmasının olumsuz bulgusu, diskteki en eksik katalogla üretilmiştir.**

Düzeltilmiş katalogla aynı pencere **1.734 pozitif (%39,9)** vermektedir; taban
oranındaki bu 15 puanlık kayma, sonucun ölçüldüğü koşullu tabanı doğrudan
değiştirmektedir.

---

## 5. Yapılan düzeltme

AFAD kataloğu, TDVMS dalga biçimi portalından **bağımsız**, kamuya açık ve kimlik
doğrulaması gerektirmeyen bir API üzerinden yayımlanmaktadır. 2000–günümüz ulusal
kataloğunun tamamı **tek bir istekle, 30 saniyenin altında** alınmaktadır. Eksik
bir kopya ile çalışmayı gerektiren teknik bir kısıt bulunmamaktaydı.

`scripts/fetch_afad_catalog.py` ile yeniden kurulmuştur:

- **576.829 olay**, M≥0, 04.01.2000 – 29.08.2026
- Önceki tüm dosyaların **kesin üst kümesi** olduğu doğrulanmıştır: hiçbir olay
  kaybedilmemiş, 19.353 olay eklenmiştir.
- Eski üç katalog silinmemiş, `catalogs/archive_superseded_2026-08-30/` altına
  arşivlenmiştir; bunlar yayımlanmış sonuçların köken kaydıdır ve o sonuçların
  yeniden üretilmesi için gereklidir.
- Tüm kod artık tek bir kanonik yola (`catalogs/catalog_current.csv`, sembolik
  bağ) bakmaktadır.

İki teknik ayrıntı kayda geçirilmiştir: 2010 öncesi API kayıtları saniyenin kesirli
kısmını içermediğinden zaman ayrıştırması karışık ISO-8601 biçimini kabul edecek
şekilde yapılmalıdır; ve sınırlayıcı kutu 44,5°K enlemine ulaşmalıdır — 43,5°
sınırı altı Karadeniz olayını sessizce dışarıda bırakmaktadır.

---

## 6. Düzeltmenin sonuçlara etkisi

Düzeltme sonuçları **iki yöne birden** taşımıştır. Bu, yapılanın bir ayarlama
değil ölçüm olduğunun en açık göstergesidir.

### 6.1 Bölge bazlı tahmin — iyileşme

Blok düzeyinde (30 günlük ayrık bloklar, 3 tohum):

| bölge | önce | sonra | fark |
|---|---|---|---|
| Ege | 0,519 | **0,692** | +0,173 |
| Orta | 0,396 | **0,618** | +0,222 |
| Doğu Anadolu | 0,662 | 0,667 | +0,005 |
| Kuzey Anadolu | 0,464 | 0,410 | −0,054 |

Kazanımlar **kusurun bulunduğu yerde** ortaya çıkmaktadır: eksik olaylar ezici
çoğunlukla Ege açıklarındaydı ve iyileşme Ege ile ona komşu Orta bölgesinde,
tohum yayılımının beş–on katı büyüklükte gerçekleşmiştir; doğuda kalan Doğu
Anadolu değişmemiştir. Bir varyans artefaktı bu coğrafyaya uymaz.

**Orta bölgesinin rastlantı düzeyinden 0,618'e çıkması, önceki raporun bu bölgeyi
"Poisson'a yakın, dolayısıyla tahmin edilemez" biçimindeki fiziksel teşhisini
geçersiz kılmaktadır.** Aynı teşhis Kuzey Anadolu için geçerliliğini korumaktadır.

### 6.2 Kaotik öznitelikler — kötüleşme

Yoğun bir artçı dizisinin geri gelmesi taban oranını %25,1'den %39,9'a çıkarmış,
kalıcılık tabanı 0,5423'ten 0,5713'e yükselmiştir. Yoğun bir artçı dizisi tam
olarak "önceki olaydan bu yana geçen gün" değişkeninin iyi kestirdiği şeydir;
dalga biçimi türevli öznitelikler bu yükselişe yetişememiş ve dört model
türevinin tamamı taban altına inmiştir.

### 6.3 Tespit — etkilenmemiştir

Katalog, dedektörün negatif (gürültü) sınıfını denetlemekte de kullanılmaktadır.
Kurulmuş veri kümeleri pencere pencere yeniden denetlenmiş, **55.595 gürültü
penceresinden 3'ünün** (%0,005) içinde katalogca kaçırılmış bir olay bulunduğu
görülmüştür. Bunlar negatif etiketlenmiş pozitiflerdir; yani hata **koruyucu
yöndedir** — model doğru ateşlediği için cezalandırılmaktadır. Yayımlanan tespit
değerleri etkilenmemekte, yeniden eğitim gerekmemektedir.

---

## 7. Sonuç ve öneri

`deprem_katalog_utc.csv` **kullanımdan kaldırılmalıdır**. Gerekçeler bağımsız
olarak yeterlidir:

- 2010 öncesi hiç kayıt içermemesi, uzun süreli katalog çalışmalarını sessizce
  geçersiz kılmaktadır (ölçülen etki: AUC 0,3098).
- Kapsadığı dönemde bile %38,8 eksiktir ve eksiklik M6,0'a kadar uzanmaktadır.
- Eksikliğin çeyrekten çeyreğe on üç kat değişmesi, herhangi bir düzeltme
  katsayısıyla telafiyi olanaksız kılmaktadır.

Dosya yalnızca **köken kaydı** olarak arşivde tutulmalı, yeni hiçbir çalışmada
kullanılmamalıdır. Yayımlanmış sonuçların yeniden üretilmesi gerektiğinde arşiv
yolundan okunmalı; bu durum ilgili belgelerde açıkça belirtilmiştir.

Daha genel çıkarım şudur: kusur aylarca fark edilmeden kalmıştır, çünkü yerel
dosya hiçbir zaman **kendi üst kaynağıyla** karşılaştırılmamıştır. Bu karşılaştırma
tek bir HTTP isteği ve yarım dakika sürmektedir ve artık bir varsayım değil,
depoda bir betiktir.

---

*Kaynaklar: `docs/experiment_neural_forecasters_2026-08-30.md`,
`docs/experiment_chaos_forecast_2026-08-27.md`,
`scripts/fetch_afad_catalog.py`, `experiments/analyses/afad_audit.py`*
