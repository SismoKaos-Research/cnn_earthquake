# 2. LİTERATÜR ÖZETİ

## 2.1 Derin Öğrenme Tabanlı Tespit ve Faz Toplama

Kısa pencereli olay/gürültü ayrımı, makine öğrenmesi tabanlı sismolojinin
yerleşik problemlerindendir. **Perol et al. (2018)**, ConvNetQuake ile ham
dalga formu üzerinde çalışan bir evrişimli ağın tespit ve kaba konumlandırmayı
birlikte yapabildiğini Oklahoma indüklenmiş depremselliğinde göstermiş; küçük
büyüklüklerde klasik şablon eşlemeye kıyasla belirgin kazanım bildirmiştir.
**Ross et al. (2018)**, Güney Kaliforniya'da analist tarafından
işaretlenmiş milyonlarca kayıttan eğitilen genelleştirilmiş faz tespiti (GPD)
yaklaşımıyla P ve S fazlarını 4 saniyelik pencerelerden doğrudan
sınıflandırmıştır; çalışma, tek istasyonda dalga formu şeklinin faz ayrımı için
yeterli olduğunu ortaya koymuştur. **Zhu ve Beroza (2019)**, PhaseNet ile faz
toplamayı bir U-Net bölütleme problemi olarak yeniden formüle etmiş ve varış
zamanı belirsizliğini olasılık dağılımı olarak modellemiştir. **Mousavi et al. (2020)**, EQTransformer ile dikkat mekanizmasını kullanarak tespit ve
faz toplamayı tek bir ağda birleştirmiş; hiyerarşik dikkat yapısının uzun
kayıtlarda seyrek olayları yakalamayı kolaylaştırdığını göstermiştir.

Bu çalışmaların ortak yönü, **tespit** görevinde derin öğrenmenin klasik
yöntemlere üstünlüğünün yerleşmiş olmasıdır. **Jover-Alfaro et al.
(2026)**, makine öğrenmesinin sismolojideki iki kullanımını ayırmanın
gerekliliğine dikkat çekmektedir: faz toplama ve katalog üretimi gibi veri
işleme uygulamaları sağlam sonuçlar verirken, doğrudan deprem öngörüsü
uygulamaları tartışmalı kalmaktadır. Bu projedeki tespit görevi birinci
kategoridedir.

Türkiye'de yürütülen çalışmalar arasında **Başar ve Çelik (2026)**, yüksek
örnekleme hızlı (5 Hz) GNSS hız zaman serilerinden sismik olay tespiti için
hibrit bir CNN–LSTM mimarisi önermiştir. Çalışma bu projeyle üç noktada
örtüşmektedir: (i) aynı mimari aile (evrişimli kol + LSTM kolu); (ii) çok
kanallı karar şemalarının (oy tabanlı, herhangi-kanal, ağırlıklı füzyon) kaçan
tespit ile yanlış alarm dengesini belirlemesi; (iii) bağımsız gerçek veri
üzerinde ölçülen başarım düşüşünün **eğitim–test dağılım kayması** ile
açıklanması. Ayrıştıkları nokta ölçüm türüdür: bu projede sismometre kaydı,
anılan çalışmada GNSS hız serisi kullanılmaktadır.

## 2.2 Mimari Kaynağı ve Alanlar Arası Aktarım

Bu projenin temel aldığı çift kanallı yapı, **Wang ve Zhao (2025)** tarafından
*Applied Soft Computing* dergisinde önerilen 1D2D-EDL mimarisidir. Anılan
çalışmanın özgün uygulama alanı sismoloji değil **rulman arıza teşhisidir**.
İki katkı sunulmaktadır: (i) tek boyutlu zaman serisini, çoklu vektörler ile bir
merkez vektör arasındaki açı farklarını hesaplayarak iki boyutlu görüntüye
dönüştüren **bağıl açı matrisi** (relative angle matrix, RAM); (ii) 1B kanalında
LSTM ve çok başlı öz-dikkatin, 2B kanalında evrişimli bileşenlerin kullanıldığı
ve iki kanalın öznitelik düzeyinde birleştirildiği topluluk ağı.

Mimarinin titreşim sinyallerinden sismik dalga formlarına aktarılması bu projenin
çıkış noktasıdır. Aktarımın sınırı ölçülmüştür: RAM dönüşümünün ölçek
değişmezliği, sismik tespitte ayırt edici gücün büyük bölümünü taşıyan **genlik**
bilgisini yapısal olarak elemektedir (Bölüm 5.2). Kaynak alanda sorun
oluşturmayan bir özelliğin hedef alanda belirleyici kısıt hâline gelmesi, alanlar
arası mimari aktarımında sistematik olarak sınanması gereken bir durumdur.

## 2.3 Karşılaştırma Tabanı ve Doğrulama Protokolü

Yayımlanan dedektörler rutin olarak 0,95 üzerinde accuracy bildirmektedir. Ancak
bu iyileşmenin ölçüldüğü taban çoğunlukla ya çoğunluk sınıfı ya da klasik bir
STA/LTA tetikleyicisidir (**Allen, 1978**). Bu çalışmanın savı, alışılmış biçimde
kurulmuş bir veri kümesi için bu tabanların yetersiz olduğudur (ayrıntı için
Bölüm 3.9).

Bu kaygı literatürde bağımsız olarak doğrulanmaktadır. **Jover-Alfaro et al. (2026)**, %97'nin üzerinde accuracy bildiren bir iş akışını Tokyo
verisiyle yeniden üretmiştir. Standart rastgele eğitim–test ayrımları bildirilen
başarımı (>%99) tekrarlamış; ancak zaman temelli doğrulama ve ileri-yürüyen
sınama uygulandığında başarım **%24'e**, doğrudan konumlar-arası sınamada
rastlantıdan ayırt edilemeyen **%16'ya** düşmüştür. Yazarlar bunu veri sızıntısına
bağlamakta ve modelin fiziksel öncüller yerine yerel yapaylıklara dayandığı
sonucuna varmaktadır.

**Stockman et al. (2026)**, EarthquakeNPP ölçüt kümesiyle aynı disiplini
öngörü tarafında kurmakta; sinirsel nokta süreçlerinin başarımının ancak yerleşik
ETAS temel modellerine karşı ölçüldüğünde anlamlı olduğunu vurgulamaktadır.
**Albelali ve Ahmed**, LSTM değerlendirmelerinde veri sızıntısının yapılandırma
ve doğrulama stratejisine göre nasıl değiştiğini belgeleyerek, bölümleme
protokolünün etkisinin model seçiminden büyük olabileceğini göstermektedir.

Bu literatür, bu projedeki üç tercihi doğrudan desteklemektedir: **istasyon-ayrık
bölümleme** (Bölüm 3.5), **koşullu genlik tabanı** (Bölüm 3.9) ve yeniden eğitim
yapılmaksızın **korpuslar arası sınama** (Bölüm 4.9).

## 2.4 Veri Kümeleri

**Mousavi et al. (2019)** tarafından derlenen STEAD, analist denetiminden
geçmiş P ve S işaretlemelerini iz başına HDF5 öznitelikleri olarak sunan küresel
bir veri kümesidir ve korpuslar arası genelleme sınamalarında fiilî standart
hâline gelmiştir. **Woollam et al. (2022)** tarafından geliştirilen
SeisBench, veri kümeleri ile modeller arasında tekrarlanabilir karşılaştırma için
ortak bir çerçeve sunmaktadır. Kuramsal varış zamanları, **Kennett ve Engdahl
(1991)** iasp91 hız modeli ile **Crotwell et al. (1999)** TauP algoritması
kullanılarak hesaplanmaktadır.

---

## 2.5 Değerlendirme Protokollerinin Karşılaştırılması

Derin öğrenmeyle P dalgası tespiti yazınında bildirilen başarım değerleri 0,69
ile 0,98 arasında değişmektedir. Bu yayılım mimariden çok **değerlendirme
protokolüne** bağlıdır; aşağıdaki çizelge bu nedenle başlıca değerden önce
protokolü karşılaştırmaktadır.

**Çizelge 1.** P dalgası tespiti yazınında değerlendirme protokolleri.

| | TransQuake (2021) | NZ edge CNN (2026) | CWT + YOLO (2025) | Bu çalışma |
|---|---|---|---|---|
| Görev | ikili: pencerede P var mı | üç sınıf: P / S / gürültü | ikili: spektrogram P mi | ikili: pencerede P var mı |
| Pencere | 50 s | 2 s | görüntü başına | 3,4 s (1,4 s P sonrası) |
| Negatifler | FilterPicker yanlış işaretlemeleri | aynı 90 s kayıttan, ±2 s dışlama | aynı istasyon kaydından | 3 sa önceki taranmış sakin pencereler, 482.898 olaylık katalogla ±300 s denetimli |
| Bölümleme | zamansal | rastgele 70/15/15 | istasyonlar arası | istasyon-ayrık |
| Test dengesi | ~11:1 | dengeli | dengeli | dengeli |
| Büyüklük | artçılar | M ≥ 3,0 | 3° içinde | M ≥ 2,0 (medyan 2,3), ≤ 56 km |
| Koşullu taban | bildirilmemiştir | bildirilmemiştir | bildirilmemiştir | 0,6679 |
| Başlıca değer | precision 0,740 / recall 0,685 | accuracy %97,12 | precision 0,934 / recall 0,942 | ROC-AUC 0,8712 / recall 0,638 |

Negatiflerin seçimini denetleyen iki çalışma recall bakımından 0,64–0,69
aralığında, denetlemeyen ikisi 0,94–0,98 aralığındadır.

Mousavi et al. (2020) ve Zhu & Beroza (2019) çalışmalarının aksine, bu dört
çalışmanın hiçbiri koşullu bir taban bildirmemektedir.

TransQuake, pencere uzunluğunu 20 s ile 50 s arasında taramış ve metriklerin
pencere uzunluğu arttıkça iyileştiğini, bunun da P dalgası dışındaki bilginin
tespite katkıda bulunduğunu gösterdiğini bildirmiştir. Aynı çalışma, farklı
episantr uzaklıkları göz önüne alındığında yalnızca tam bir P dalgası içeren
sabit bir zaman penceresi belirlemenin mümkün olmadığını da kaydetmektedir.

NZ edge CNN çalışması zamansal bölümlemenin farklı dönemler arasında gözlenen
sapmalar nedeniyle başarısız olduğunu ve bu nedenle rastgele bölümlemeye
geçildiğini bildirmektedir; gürültü kesitleri P ve S kesitleriyle aynı 90 s
kayıttan yalnızca ±2 s dışlama ile çekilmektedir. Çalışmanın gömülü donanım
katkısı (~38 bin parametre, 7 ms altı çıkarım) bu karşılaştırmanın kapsamı
dışındadır.

# 3. GEREÇ VE YÖNTEM

## 3.1 Veri

Olaylar **AFAD ulusal kataloğundan** alınmıştır. *(Düzeltme, 31.08.2026: bu
bölüm daha önce kataloğu KRDAE/KOERI'ye atfediyordu. 30.08.2026'da doğrulandığı
üzere yerel katalog dosyalarındaki her EventID bir AFAD eventID'sidir; büyüklük
ve koordinatlar AFAD API'siyle basılan basamağa kadar aynıdır. **Dalga
biçimleri** KOERI'dendir — KO ağı, KOERI FDSN servisi, aşağıda — yanlış olan
yalnızca katalog atfıydı.)* İki katalog dosyası farklı amaçlarla
kullanılmıştır: 93.690 olaylık dosya indirme listesi, 482.898 olaylık dosya ise
yalnızca gürültü taraması için. Tarama kataloğunun çok daha küçük büyüklüklere
kadar tam olması (medyan M 1,70; %32,9'u M 1,5 altında) gürültü sınıfının
güvenilirliği açısından belirleyicidir.

Dalga formları, her episantrın **0,5° (~55 km)** yarıçapındaki tüm `HH*`
kanalları (100 Hz, yüksek kazançlı geniş bant) için KOERI FDSN servisinden talep
edilmiş; her olay için başlangıç zamanından itibaren 60 s kayıt alınmıştır.
Erişilen olay dosyası **33.795**, temsil edilen istasyon **183**'tür
(KO ×156, 6G ×17, IJ ×8; iki istasyonun başlık kaydında ağ kodu
bulunmamaktadır).

**Çizelge 2.** Korpusa giren olayların özellikleri.

| Özellik | Değer |
|---|---|
| Büyüklük — medyan (p5 / p95 / maks) | 2,30 (2,00 / 3,40 / 7,70) |
| M 2,5 altındaki oran | %63,5 |
| Derinlik — medyan (p90) | 7,0 km (12,2 km) |
| Katalog konum RMS kalıntısı — medyan (p90) | 0,42 s (0,72 s) |
| Episantr uzaklığı — medyan (p95 / maks) | 38,6 km (53,5 / 55,6 km) |

Konum RMS kalıntısı, Bölüm 3.3'teki kuramsal varış zamanlarının doğruluğuna üst
sınır getirdiği için raporlanmaktadır.

Gürültü pencereleri, her olayın başlangıç zamanından **3 sa 05 dk – 3 sa 00 dk**
öncesindeki 300 s kesitten çekilmiş; her aday 482.898 olaylık katalogla
denetlenerek ±300 s içinde katalog kaydı bulunanlar elenmiştir.

> **Denetim kataloğunun eksikliği ve ölçülen etkisi (31.08.2026).** Denetimde
> kullanılan katalog, bölgeye ait AFAD olaylarının ~%29'unu — Şubat 2025
> Santorini–Amorgos dizisinin neredeyse tamamı dâhil — içermemektedir. Negatif
> sınıf tam olarak bu katalogla doğrulandığından, kurulmuş veri kümeleri
> yeniden kurulan katalogla pencere pencere denetlenmiştir. Her gürültü
> penceresinin mutlak zamanı dosya adından geri hesaplanabilmektedir
> (`noise_event_<id>_..._win<k>` → köken − 3 sa 05 dk + 1,7k s):
>
> | küme | gürültü penceresi | pencere **içinde** olay | ±300 s (denetim eşdeğeri) |
> |---|---|---|---|
> | `ponly_3p4s` (3,4 s) | 55.595 | **3 (%0,005)** | 2.260 (%4,07) |
> | `catalog_6s_hard` (6 s) | 55.568 | **32 (%0,058)** | 2.185 (%3,93) |
>
> Yalnızca orta sütun yanlış etiketlemedir; ±300 s sütunu denetimin güvenlik
> payıdır — 300 s uzaktaki bir olay 3,4 s'lik pencerede bulunmamaktadır. Eski
> katalog her iki sütunda da **sıfır** işaretlemektedir, yani denetim eldeki
> katalogla doğru çalışmıştır; kaçanların tümü göremediği olaylardır. Dahası
> hata **koruyucu yöndedir**: bunlar negatif etiketlenmiş pozitiflerdir, model
> doğru ateşlediği için cezalandırılmaktadır. **55.595'te 3 pencerede hiçbir
> raporlanan tespit değeri etkilenmemekte, yeniden eğitim gerekmemektedir.** Denetim yalnızca
zamansaldır; bu, bilinçli olarak aşırı temkinli bir tercihtir. Kullanılabilir
gürültü sinyalden yaklaşık 50 kat fazladır (1.784.650'ye karşılık 35.836
pencere); bu asimetri Bölüm 3.6'teki madenciliği ek maliyet olmaksızın mümkün
kılmaktadır.

## 3.2 Sinyal İşleme

Her iki sınıf ve her iki korpus tek bir uygulamadan geçmektedir. Bileşen bazında:
(1) doğrusal eğilim giderme, (2) sabit eğilim giderme, (3) baş ve sondaki %5'e
Hann pencereleme, (4) 1–45 Hz 4. derece Butterworth bant geçiren süzgeç,
`filtfilt` ile (sıfır fazlı; varışta grup gecikmesi kayması oluşmaz),
(5) 100 Hz'e polifaz yeniden örnekleme. Bileşen seçimi alfabetik kanal koduna
göre değil **role göre** (Z, sonra N/1, sonra E/2) yapılmakta; böylece karma
sensör kodlu istasyonların düşey bileşensiz veri kümesine girmesi
engellenmektedir.

Her pencere `{seq, img}` ikilisi üretmektedir: `seq` (600, 3) boyutunda
standartlaştırılmış üç bileşenli dalga formu, `img` ise (3, 129, 10) boyutunda
log-güç STFT'dir (n_fft = 256, hop = 64, top_db = 80). Boyutlar 6 s
yapılandırması içindir.

**Genlik normalizasyonu.** İki kanal farklı normalize edilmektedir. `img` için
her istasyonun kendi gürültü kayıtlarından hesaplanan, frekans bandı başına
medyan dB profili çıkarılmakta; sonuç *istasyonun kendi gürültü tabanı üzerindeki
desibel* değeridir, dolayısıyla enstrüman kazancı sadeleşirken gerçek genlik
korunmaktadır. `seq` için her bileşen istasyonun uzun dönemli gürültü ortalaması
ve standart sapmasına göre standartlaştırılmaktadır.

Bu ayrım işlevseldir. Pencere kendi istatistiklerine göre standartlaştırıldığında
her örnek zorunlu olarak 0 ortalama ve 1 standart sapmaya çekilmekte, mutlak
genlik bilgisi tamamen silinmektedir: bu durumda `seq` standart sapmasının
ROC-AUC değeri tam olarak 0,5000 (rastlantı), istasyon referansı altında ise
0,9440 ölçülmüştür. Referans kapsamı 177 istasyonda 531 (istasyon, bileşen)
çiftidir.

**İki STFT geometrisi.** Yukarıdaki boyutlar `n_fft = 256, hop = 64` ile
üretilen ilk kümeye aittir. Bölüm 4'te bildirilen sonuçların alındığı kümeler
ise `n_fft = 64, hop = 16` ile üretilmiş olup `img` boyutu 6 s için
(3, 33, 38), 3,4 s için (3, 33, 22)'dir. Kısa pencerede büyük bir FFT
penceresi neredeyse hiç zaman karesi bırakmadığından bu değişiklik
gereklidir; etkisi Bölüm 4.8'de 0,0010 olarak ölçülmüştür.

## 3.3 Varış Sabitleme

Varışlar toplanmamakta, **kestirilmektedir**. Her (olay, istasyon) çifti için
episantr uzaklığı katalog hiposantrı ve istasyon koordinatlarından hesaplanmakta;
ilk varan P fazı (`p`, `P`, `Pg`, `Pn`) iasp91 ile TauP kullanılarak
belirlenmektedir. Pencere, kestirilen varıştan pencere uzunluğunun üçte biri
kadar önce başlatılmaktadır (6 s için 2,0 s, 3 s için 1,0 s). **Hiçbir
tetikleyici ve eşik uygulanmamaktadır**; sessiz olduğu için elenen kayıt yoktur.

**Çizelge 3.** Kestirilen varışların, varışı görebilecek kadar kısa bir LTA ile
(STA 0,2 s / LTA 1,0 s) yeniden hesaplanan işaretlemelere karşı doğrulanması.

| Ölçüt | Değer |
|---|---|
| Medyan kalıntı (işaretleme − kestirim) | +0,84 s |
| Medyan mutlak sapma | 0,63 s |
| ±2 s içinde kalan oran | %75,7 |

Kalıntının pozitif olması beklenmektedir; bir tetikleyici gerçek başlangıcın
gerisinde kalır. Bağımsız ikinci denetim olarak, varış sonrası RMS'in varış
öncesi RMS'i aşma oranı düşey bileşenlerde %96,8, medyan oran 8 kattır. Bu
doğruluk **tespit için yeterli, başlangıç zamanı kestirimi için yetersizdir**;
veri kümesi faz toplama amacıyla kullanılmamalıdır. Kayıt düzeyinde tutulma
oranı **%96,3**'tür (55.568 istasyon kaydı).

**S varışı üretimde hesaplanmamaktadır.** Faz listesi
`["p", "P", "Pg", "Pn"]` ile sınırlıdır; S varışının pencere içine düşüp
düşmediği üretim aşamasında denetlenmemektedir. Aynı (uzaklık, derinlik)
çiftleri üzerinde iasp91 ile sonradan hesaplandığında 6 s penceresi için
aşağıdaki tablo elde edilmektedir.

**Çizelge 4.** 6 s penceresinde S varışının konumu (pencere `[P − 2 s, P + 4 s]`).

| Uzaklık | Kayıt | S pencere içinde | Medyan S−P |
|---|---|---|---|
| 0–25 km | 10.647 | %99,3 | 2,56 s |
| 25–50 km | 35.074 | %15,5 | 5,11 s |
| 50–100 km | 9.847 | %0,0 | 6,70 s |
| Tümü | 55.568 | %28,8 | 5,09 s |

Tespit test bölümünde aynı oran %32,5'tir. S−P uzaklıkla ölçeklendiğinden S
varlığı uzaklığa, dolayısıyla SNR'ye koşuttur; bu üçü bu korpusta birbirinden
ayrıştırılamamaktadır.

## 3.4 Pencere Geometrisi: Yalnızca P

Bölüm 3.3'teki ölçüm nedeniyle ikinci bir pencere geometrisi kurulmuştur:
`[P − 2,0 s, P + 1,4 s]`, toplam 3,4 s.

**Ön tampon pencere uzunluğuyla ölçeklenmemektedir.** Üreticinin varsayılanı
`pencere / 3` olup 3,4 s için 1,13 s vermektedir; kestirilen varışın 0,63 s
medyan mutlak sapması karşısında bu, başlangıcın pencere dışına düşmesine yol
açmaktadır. Ön tampon 2,0 s'de sabit tutulduğunda kayıt düzeyinde tutulma
**%96,4** olup 6 s yapılandırmasının %96,3 değeriyle eşleşmektedir. Üretilen
olay dosyası 32.880, istasyon kaydı 55.595'tir.

Üretilen kayıtların tamamı iasp91 ile yeniden denetlenmiştir: **S varışının
pencereye girdiği kayıt bulunmamaktadır.** En küçük S−P 1,450 s olup kesme
noktasına payı +0,050 s'dir.

Bu güvence **hız modeline görelidir.** S−P katalog hiposantrından
kestirildiğinden katalogun konum hatasını taşımaktadır (medyan RMS kalıntısı
0,42 s; 5 km'lik bir uzaklık hatası S−P'yi ~0,6 s kaydırmaktadır).

**Çizelge 5.** S−P kestirim hatasına göre risk altındaki kayıt sayısı.

| S−P kestirimi şu kadar saparsa | S içerebilecek kayıt |
|---|---|
| 0,00 s | 0 |
| 0,30 s | 740 (%1,3) |
| 0,50 s | 1.445 (%2,6) |
| 0,63 s | 2.021 (%3,6) |

Bu nedenle sonuç "S içermez" olarak değil, **"iasp91 altında yalnızca P"**
olarak bildirilmektedir.

## 3.5 Bölümleme

Bölümler **istasyon-ayrıktır**: her istasyon, her iki sınıf için de tek bir bölüme
atanmaktadır; eğitimde görülen bir istasyonun test kümesinde herhangi bir etiket
altında yeniden görünmesi yapısal olarak olanaksızdır. Atama tohumlanmış olup
üretim tekrarlanabilirdir.

**Çizelge 6.** İstasyon-ayrık bölümleme (6 s).

| Bölüm | İstasyon | Sınıf başına pencere |
|---|---|---|
| Eğitim | 120 | 38.247 |
| Doğrulama | 28 | 9.415 |
| Test | 35 | 7.906 |

Eğitim ∩ Test kesişimi boş kümedir. Bölümlemenin enstrüman kazancı sızdırmadığı
ayrıca sınanmıştır: `img` ortalama dB istatistiği havuzlanmış olarak 0,9205, test
istasyonlarının *içinde* hesaplanıp örnek ağırlıklı alındığında 0,9221
vermektedir. Genlik sinyali tekil istasyonların içinde de varlığını sürdürmekte,
yani ezberlenmiş bir istasyon parmak izi değildir.

## 3.6 Zor Negatif Madenciliği

Gürültü sinyalden ~50 kat bol olduğundan hangi gürültünün kullanılacağı bir
tasarım tercihidir. Her aday pencere, kendi (istasyon, bileşen) gürültü sigmasına
göre en yüksek bileşeni üzerinden sıralanmakta ve gereken sayı **%75–%99
bandından** eş aralıklı çekilmektedir. İki karar belirleyicidir:

- **Sıralama küresel olmalıdır.** Dosya içinde sıralayan ilk uygulama tabanı
  yalnızca 0,9535'ten 0,9312'ye indirmiştir; tek bir 300 s dosyasındaki pencereler
  aynı istasyon ve saati paylaştığından neredeyse eşit gürültüdedir. Genlik
  değişkenliğinin neredeyse tamamı istasyonlar ve zamanlar *arasındadır*.
- **Üst sınırın %99'da tutulması bilinçlidir.** Taranmış bir arşivin en gürültülü
  ucu, katalogca kaçırılmış bir depremin saklanacağı yerdir; oradan seçim yapmak
  negatif sınıfa pozitif örnek katma riski taşır.

Madenlenmiş ve madenlenmemiş kümeler olaylar, bölümler ve istasyon ataması
bakımından birebir aynıdır; yalnızca hangi gürültü pencerelerinin tutulduğu
değişmektedir. Bu, ikisini kontrollü bir çift hâline getirmektedir.

**Dört negatif rejimi.** Yalnızca P kümesi üzerinde, aynı pozitif pencereler
korunarak dört farklı negatif seçimi kurulmuştur. Dördü de aynı 35 test
istasyonunu ve aynı 7.908 olay penceresini paylaşmakta, yalnızca hangi gürültü
pencerelerinin tutulduğu değişmektedir.

| Rejim | Tanım |
|---|---|
| `matched` | Negatif genlik **dağılımı** pozitiflerinkini yansıtmaktadır |
| `band` | Havuzun %75–99 dilimi (yalnızca yüksek genlik) |
| `wideband` | %99 altındaki tüm havuz, dilimler arasında eşit yayılımlı |
| `natural` | Madencilik yok; havuzun kendi yoğunluğu |

**Genlik eşleştirmesinin gerekçesi.** %75–99 bandı, her negatifin altına
pozitiflerde bulunmayan bir genlik tabanı koymaktadır. 6 s penceresinde bu
zararsızdır: olaylar S ve kodayı taşıdığından yüksek genliklidir. Yalnızca P
penceresinde değildir; sonucu Bölüm 4.1'de verilmektedir.

Eşleştirme, merkezî bir değere değil **dağılımın tümüne** yapılmaktadır:
ortalamaları eşit, yayılımları farklı iki sınıfta "her iki yönde de uç değer"
ayırt edici kalmaktadır. Eşleştirme havuz derinliğiyle sınırlıdır (Bölüm 5.8).

## 3.7 Model ve Eğitim

Bağımsız kol ablasyonlarına izin veren çift kollu bir ağ kullanılmıştır.
**2B kolu** spektrogram üzerinde üç evrişim aşamasıdır
(Conv 3→32 → Conv 32→64, adım 2 → Conv 64→128, adım 2; her aşamada BatchNorm ve
GELU, ardından AdaptiveAvgPool ile 128 boyut). **1B kolu**, ham dalga formu
üzerinde çift yönlü LSTM (gizli 48) ve 4 başlı öz-dikkatten oluşup artık bağlantı
ve zaman ekseninde ortalama ile 96 boyut üretmektedir. Etkin her kol 96 boyuta
izdüşürülmekte; birleştirme ya iki öğrenilen skalerle doğrusal
(F = a·F₁ + b·F₂) ya da örnek başına kapı ile (F = g(x)·F₁ + (1−g(x))·F₂)
yapılmaktadır. Baş, `LayerNorm → Dropout → Linear(96) → GELU → Dropout →
Linear(1)` biçiminde tek lojit üretmektedir.

**Çizelge 7.** Parametre sayıları ve eğitim yapılandırması.

| Yapılandırma | Parametre | Örnek başına |
|---|---|---|
| Yalnız 2B | 115.459 | 1,5 |
| Yalnız 1B | 76.707 | 1,5 |
| İkisi, kapılı birleştirme | 191.874 | 3,8 |

Eğitim: `BCEWithLogitsLoss` (etiket yumuşatma 0→0,1 / 1→0,9), AdamW
(lr 2×10⁻⁴, ağırlık sönümü 3×10⁻²), kosinüs tavlama, gradyan kırpma 1,0, yığın
32, seyreltme 0,4, azami 80 dönem, doğrulama ROC-AUC 10 dönem sabit kaldığında
erken durdurma, en iyi doğrulama ROC-AUC ağırlıkları. Her yapılandırma 42, 43, 44
tohumlarıyla eğitilmekte ve olasılıklar ortalanmaktadır; **tohumlar arası
yayılım birincil güvenilirlik istatistiğidir**.

## 3.8 Değerlendirme Ölçütleri

Birincil ölçüt ROC-AUC'dir; eşikten bağımsız olduğu için sabit 0,5 kesme noktası
kullanan bir modelle eşik seçimi yapan bir tabanın adil karşılaştırılmasına izin
vermektedir. Ayrıca accuracy, MCC, PR-AUC ve Brier skoru raporlanmaktadır.

## 3.9 Karşılaştırma Tabanları

Bir başarım değeri, tek başına yorumlanamaz; ancak görevin *modelsiz* olarak ne
kadarının çözülebildiğine kıyasla anlam kazanır. Bu çalışmada üç farklı taban
türü ayrılmaktadır.

**(a) Çoğunluk sınıfı tabanı.** Her zaman en sık görülen sınıfı tahmin eden
kuraldır. Dengelenmiş bir veri kümesinde bu kural tanım gereği 0,5000 ROC-AUC
vermektedir; yani hiçbir bilgi taşımaz. Literatürde yaygın olarak kullanılmasına
karşın, dengeli bir ölçüt kümesinde **boş bir tabandır** ve modelin katkısını
sistematik olarak abartmaktadır.

**(b) Klasik yöntem tabanı (STA/LTA).** Kısa dönem ortalamasının uzun dönem
ortalamasına oranını eşikleyen geleneksel tetikleyicidir (Allen, 1978). Bu
çalışmada, model ile **birebir aynı pencereler** üzerinde, kaynak MiniSEED
verisinden dosya/istasyon/pencere indeksi ile yeniden kurularak hesaplanmıştır.
Parametreler doğrulama bölümünde seçilmiş, test yalnızca bir kez
değerlendirilmiştir.

**(c) Koşullu genlik tabanı.** Bu çalışmanın önerdiği ve tüm karşılaştırmalarda
esas aldığı tabandır. Hiçbir öğrenme içermeksizin, doğrudan saklanmış
tensörlerden okunan **tek bir skaler istatistiğin** ulaştığı ROC-AUC değeridir.
Gerekçesi veri kümesinin kuruluş biçimidir: negatifler bilinçli olarak sakin bir
aralıktan, pozitifler ise varış merkezinden çekildiği için iki sınıf modelleme
öncesinde yaklaşık 19 dB farklıdır. Dolayısıyla "bu pencere ne kadar gürültülü"
sorusu, öğrenen bir dedektörün fiilen aşması gereken engeldir.

Üç istatistik hesaplanmaktadır: `seq` mutlak maksimumu, `seq` standart sapması ve
`img` ortalama dB değeri. En yüksek değeri veren istatistik **taban** olarak
alınmakta ve modelin **katkısı** şu şekilde tanımlanmaktadır:

katkı = topluluk ROC-AUC − taban

Taban değerleri **yönlendirilmiş** olarak, yani max(a, 1−a) biçiminde
bildirilmektedir; ters yönde kestirim yapan bir kural da en az onun kadar
sömürülebilir olduğundan bu düzeltme gereklidir.

**Tabanlar farklı olduğunda katkılar doğrudan karşılaştırılamaz**, çünkü daha
düşük bir taban kazanılacak daha fazla pay bırakmaktadır. Yapılandırmalar arası
karşılaştırmalarda bu nedenle **kazanılan pay oranı** kullanılmaktadır:

kazanılan pay = (ROC-AUC − taban) / (1 − taban)

---

**Tabanın kendisi de yanlış ölçülebilir.** ROC-AUC yalnızca monotonik
sıralamayı ölçmektedir. Bir tekil istatistik ile sınıf arasındaki ilişki
monotonik değilse — örneğin U biçimliyse — ROC-AUC, o istatistiğin
öğrenilebilir kıldığı ayrımı eksik bildirmektedir. Bu nedenle her taban iki
biçimde hesaplanmaktadır:

- **Monotonik taban:** istatistiğin yönlendirilmiş ROC-AUC değeri.
- **Monotonik olmayan taban:** aynı tekil istatistik üzerinde, eğitim
  bölümünde uyarlanıp test bölümünde değerlendirilen 4 derinlikli bir karar
  ağacının ROC-AUC değeri.

İkisi arasındaki fark, kurulumun bir yapaylık taşıyıp taşımadığının
göstergesidir. Raporda taban bildirilen tüm kümelerde bu fark ölçülmüştür.

## 3.10 Karma Duyarlıkta Genlik Kanalının Sayısal Güvenliği

`seq` kanalı istasyonun kendi gürültü sigmasının katları cinsindendir ve bu
dağılımın kuyruğu çok uzundur: katalog sabitli 6 s ölçüt kümesinde en büyük
değer 3,57 × 10⁵ iken fp16 biçimi 65.504'te doymaktadır. Mixed precision ile
eğitimde (AMP) **girdi**, herhangi bir katman çalışmadan önce fp16'ya
dönüştürülmekte, dolayısıyla 95.324 pencerenin yedisi `inf` değerine gitmekte
ve tek bir yığın tüm ağırlıkları kalıcı olarak `NaN` yapmaktadır. Sorun,
yalnızca `--channels 2d` çalıştırıldığı sürece görünmez kalmıştır; çünkü
spektrogram kolu `seq` kanalına hiç dokunmamaktadır.

Çözüm olarak işaretli logaritmik sıkıştırma uygulanmaktadır:
asinh(x) = ln(x + √(x²+1)). Sıfır civarında doğrusal, kuyrukta logaritmiktir ve
negatif değerler için tanımlıdır — dalga biçimleri sıfır ortalamalı salınımlar
olduğundan `log` bu amaçla kullanılamaz. En büyük değer 3,57 × 10⁵ → 13,48'e
inmektedir.

Dönüşümün bu çalışmada güvenli olmasının nedeni **kesin biçimde monotonik**
olmasıdır (ham ve dönüşmüş kanal arasında Spearman ρ = 1,000000). ROC-AUC sıra
temelli bir ölçüt olduğundan monotonik bir dönüşüm `seq` mutlak maksimum
tabanını değiştiremez; taban 0,9049'da kalmakta ve dönüşüm öncesi/sonrası tüm
sonuçlar aynı ölçüte göre okunabilmektedir. Pencere bazlı standartlaştırma ise
taşmayı gidermekle birlikte mutlak genliği — yani tabanın kendisini —
sileceğinden bir çözüm değildir.

Dönüşümün eniyileme davranışı üzerinde etkisi vardır: asinh altında yalın
BiLSTM 25 dönemde erken durmuşken, dönüşümsüz bir önceki koşuda 45. dönemde
hâlâ iyileşmekteydi. Dönüşüm genlik **sıralaması** bakımından etkisizdir,
eğitim dinamiği bakımından değil.
# 4. BULGULAR

Bu bölüm iki pencere geometrisinin sonuçlarını ayrı ayrı bildirmektedir.
**Yalnızca P yapılandırması (3,4 s) birincil kurulumdur**; 6 s yapılandırması,
onu ortaya çıkaran ölçümlerin kurulumu olarak Bölüm 4.8'de verilmektedir. İki
kurulum farklı sorular yanıtlamaktadır ve doğrudan karşılaştırılmaları
yanıltıcıdır: tabanları farklıdır.

Her satırda bildirilen katkı, o satırın kendi koşullu tabanına göredir.
**Kazanılan pay**, taban ile 1,0 arasındaki açıklığın kapatılan oranıdır ve
farklı tabanlara sahip kurulumları karşılaştırmanın tek doğru yoludur.

**Çizelge 8.** Başlıca sonuçlar.

| Sonuç | Değer | Taban | Katkı | Kazanılan pay | Bölüm |
|---|---|---|---|---|---|
| **Yalnızca P — birleştirme, genlik eşleştirilmiş** | **0,8762** | 0,6679 | +0,2083 | **%62,7** | 4.3 |
| Yalnızca P — yalnız 1B | 0,8712 | 0,6679 | +0,2033 | %61,1 | 4.3 |
| Yalnızca P — yalnız 2B | 0,8602 | 0,6679 | +0,1923 | %57,9 | 4.3 |
| Yalnızca P — birleştirme, doğal gürültü | 0,8410 | 0,7878 | +0,0532 | %25,1 | 4.4 |
| *Önceki kurulum:* 6 s birleştirme, zor negatif | *0,9908* | *0,9049* | *+0,0859* | *%90,3* | 4.8 |

Son satır farklı bir pencere geometrisine aittir ve yukarıdakilerle aynı
soruyu yanıtlamamaktadır.

## 4.1 Koşullu Tabanların Ölçülmesi

**Çizelge 9.** Yalnızca P kümesinde negatif rejimine göre koşullu tabanlar
(aynı pozitifler, n = 15.816).

| Rejim | Monotonik | Monotonik olmayan | Fark |
|---|---|---|---|
| `matched` | 0,6679 | 0,6658 | −0,0021 |
| `band` | 0,6447 | **0,7461** | **+0,1015** |
| `wideband` | 0,7927 | 0,7845 | −0,0082 |
| `natural` | 0,7878 | 0,7795 | −0,0082 |

`band` rejiminde `P(olay | genlik)` genlik desilleri boyunca
0,67 · 0,41 · 0,32 · 0,29 · 0,30 · 0,33 · 0,30 · 0,50 · 0,88 · 1,00
değerlerini almaktadır: en sessiz desilin %67'si olaydır. Genlik
eşleştirmesinden sonra aynı dizi
0,40 · 0,38 · 0,39 · 0,35 · 0,41 · 0,42 · 0,45 · 0,53 · 0,70 · 0,96
olmaktadır.

**Çizelge 10.** Aynı denetimin raporda taban bildirilen diğer kümelere
uygulanması.

| Küme | Monotonik | Monotonik olmayan | Fark |
|---|---|---|---|
| 6 s katalog + zor negatif | 0,9049 | 0,9003 | −0,0045 |
| 3 s katalog + zor negatif | 0,8481 | 0,8445 | −0,0036 |
| 6 s özgün (kapılı) | 0,7088 | 0,7086 | −0,0002 |
| 6 s katalog, rastgele gürültü | 0,9535 | 0,9477 | −0,0058 |
| 3 s katalog | 0,9308 | 0,9176 | −0,0132 |

Fark yalnızca `band` rejiminde 0,02'yi aşmaktadır; diğer kümelerde negatiftir.

Yalnızca P kümesinde olay ve gürültü medyanları `seq` mutlak maksimumunda 7,48
ve 3,71 (2,0 kat); 6 s kümesinde 44,38 ve 1,54'tür (~29 kat). Olayların
%35,2'si medyan gürültü penceresinden sessizdir. `img` ortalama dB tabanı
0,5010'dur.

## 4.2 6 s Penceresinde S Varışının Katkısı

Kestirilen S varışından itibaren tüm örnekler sıfırlanarak mevcut ağırlıklarla
yeniden puanlanmıştır. Kuyruk sıfırlamanın kendisi de sinyal kaldırdığından, S
içermeyen pencerelere aynı uzunluk dağılımından çekilen kuyruklar
sıfırlanarak süre eşleştirmeli bir denetim kurulmuştur.

**Çizelge 11.** S maskeleme ve süre eşleştirmeli denetim (1B kolu, recall,
eşik 0,5).

| Küme | n | Maskesiz | Maskeli | Değişim |
|---|---|---|---|---|
| S içeren → S'den itibaren sıfırlanmış | 2.567 | 0,9747 | 0,9459 | −0,0288 |
| S içermeyen → dokunulmamış | 5.339 | 0,9358 | 0,9358 | +0,0000 |
| S içermeyen → aynı uzunlukta kuyruk sıfırlanmış | 5.339 | 0,9358 | 0,9002 | −0,0356 |

Maske, maskesiz pencerelerle eğitilmiş bir model için dağılım dışıdır;
bildirilen düşüş bu nedenle bir üst sınırdır.

## 4.3 Yalnızca P: Alan İçi Başarım

**Çizelge 12.** Genlik eşleştirilmiş yalnızca P kümesinde başarım
(taban 0,6679, n = 15.816).

| Kol | Tohum başına | Ort. | Std | Topluluk | Katkı | Kazanılan pay |
|---|---|---|---|---|---|---|
| 1B (evrişim + BiLSTM) | 0,8673 / 0,8709 / 0,8671 | 0,8684 | 0,0017 | 0,8712 | +0,2033 | %61,1 |
| 2B | 0,8605 / 0,8610 / 0,8544 | 0,8586 | 0,0030 | 0,8602 | +0,1923 | %57,9 |
| **Birleştirme (doğrusal)** | 0,8730 / 0,8746 / 0,8737 | **0,8738** | 0,0007 | **0,8762** | +0,2083 | **%62,7** |

Birleştirme her iki kolu da geçmektedir. Aynı örüntü 6 s zor negatif kümesinde
de görülmüştür (Bölüm 4.8).

## 4.4 Negatif Rejimleri Arası Aktarım

Aynı modeller, aynı pozitifler üzerinde kurulmuş dört negatif rejiminde
puanlanmıştır.

**Çizelge 13.** Kazanılan pay: eğitim rejimi × değerlendirme rejimi.

| Eğitim | Kol | `matched` | `band` | `wideband` | `natural` |
|---|---|---|---|---|---|
| `matched` | 1B | %61,1 | %62,9 | %11,7 | %13,6 |
| `matched` | 2B | %57,9 | %53,4 | %14,4 | %16,2 |
| `matched` | Birleştirme | %62,7 | %65,4 | %14,4 | %16,0 |
| `natural` | 1B | %46,2 | %36,3 | %19,0 | %20,7 |
| `natural` | 2B | %33,9 | %12,1 | %20,9 | %23,1 |
| `natural` | Birleştirme | %43,0 | %30,1 | %22,8 | **%25,1** |

**Çizelge 14.** Aynı hücrelerin ROC-AUC değerleri.

| Eğitim | Kol | `matched` | `band` | `wideband` | `natural` |
|---|---|---|---|---|---|
| `matched` | Birleştirme | 0,8762 | 0,9121 | 0,8225 | 0,8217 |
| `natural` | Birleştirme | 0,8107 | 0,8226 | 0,8400 | 0,8410 |

`natural` rejiminde birleştirilmiş modelin yanlış alarm sayısı `natural`
eğitiminde 157, `matched` eğitiminde 317'dir; recall her iki modelde de
~0,633'tür.

Recall bu çizelgelerde yer almamaktadır: pozitifler tüm rejimlerde özdeş
olduğundan recall yapısal olarak değişememektedir; negatif rejimi yalnızca
yanlış alarmları değiştirmektedir.

## 4.5 Genlik Sabit Tutulduğunda Ayrım

Bir tabanı geçmek, modelin genlikten başka bir şey kullandığını
göstermemektedir: aynı skalerin daha iyi biçimlendirilmiş bir işlevi de tabanı
geçebilir (Bölüm 4.1'de bu farkın 0,1015 ettiği ölçülmüştür). Bu nedenle
genlik sabit tutularak ayrımın sürüp sürmediği ölçülmüştür.

**Çizelge 15.** Genlik deseli içinde ROC-AUC (birleştirme, `matched` eğitimli,
`matched` kümesi; havuz AUC 0,8763).

| Desil | n | `P(olay)` | Genlik aralığı | Genişlik | Desil içi AUC |
|---|---|---|---|---|---|
| 1 | 1.582 | 0,40 | 0,00 – 0,98 | 508,7× | 0,5664 |
| 2 | 1.581 | 0,38 | 0,98 – 1,51 | 1,5× | **0,6298** |
| 3 | 1.582 | 0,39 | 1,51 – 2,15 | 1,4× | **0,7090** |
| 4 | 1.581 | 0,35 | 2,15 – 3,01 | 1,4× | **0,7781** |
| 5 | 1.582 | 0,41 | 3,01 – 4,34 | 1,4× | **0,8013** |
| 6 | 1.581 | 0,42 | 4,34 – 6,43 | 1,5× | **0,8578** |
| 7 | 1.582 | 0,45 | 6,43 – 10,72 | 1,7× | **0,9274** |
| 8 | 1.581 | 0,53 | 10,72 – 21,85 | 2,0× | **0,9689** |
| 9 | 1.582 | 0,70 | 21,85 – 65,78 | 3,0× | 0,9902 |
| 10 | 1.582 | 0,96 | 65,78 – 34.794 | 528,9× | 0,9793 |

Kalın yazılan yedi desilde genlik en çok 2,5 kat değişmektedir; bu desillerin
medyanı **0,8013**'tür. 1., 9. ve 10. desillerde genlik 3–530 kat
değiştiğinden bu desiller ölçüme dâhil edilmemiştir.

`natural` eğitimli modelin `natural` kümesindeki dar desil medyanı 0,7167,
`matched` eğitimli modelin aynı kümedeki dar desil medyanı 0,7175'tir.

Sınıf dengesi desiller arasında 0,35–0,96 arasında değişmektedir; ROC-AUC
sınıf dengesinden bağımsız olduğundan bu ölçümü etkilememektedir.
## 4.6 İşletim Zarfı: Recall Neye Bağlıdır?

Yalnızca P kümesinde genel recall **0,6380**'dir (5.043 / 7.905; 2.862
kaçırılan olay, eşik 0,5).

**Çizelge 16.** Kaynak parametrelerine göre recall (1B, `matched`).

| Büyüklük | n | Recall | | log SNR | n | Recall | | Uzaklık | n | Recall |
|---|---|---|---|---|---|---|---|---|---|---|
| 2,0 | 1.611 | 0,5736 | | < 0,72 | 2.629 | 0,4690 | | 0–25 km | 1.587 | 0,6616 |
| 2,0–2,5 | 4.151 | 0,6201 | | 0,72–3,42 | 4.261 | 0,7024 | | 25–50 km | 5.044 | 0,6316 |
| 2,5–3,0 | 1.445 | 0,7031 | | 3,42–6,13 | 921 | 0,8230 | | 50–100 km | 1.274 | 0,6334 |
| 3,0–3,5 | 498 | 0,7369 | | > 6,13 | 55 | 0,9455 | | | | |
| > 3,5 | 200 | 0,8100 | | | | | | | | |

**Çizelge 17.** Kaçırılan ve bulunan olayların medyan kaynak parametreleri.

| | Kaçırılan (n = 2.862) | Bulunan (n = 5.043) | Fark |
|---|---|---|---|
| log SNR | 0,722 | 1,728 | 1,006 |
| Büyüklük | 2,200 | 2,300 | 0,100 |
| Uzaklık (km) | 39,33 | 38,77 | 0,56 |

Aynı uzaklık dilimleri 6 s yapılandırmasında 0,9773 / 0,9417 / 0,9388
vermektedir.

## 4.7 Olasılık Kalibrasyonu ve Eşik

**Çizelge 18.** Kalibrasyon ölçütleri (yalnızca P, `matched`, n = 15.816).

| | ECE | MCE | Brier |
|---|---|---|---|
| Kalibrasyon öncesi | **0,0484** | 0,3009 | **0,1386** |
| Kalibrasyon sonrası (T = 0,6008) | 0,0765 | 0,1418 | 0,1444 |

ECE ve Brier kalibrasyon sonrasında yükselmektedir. 6 s yapılandırmasında aynı
işlem ECE'yi 0,0863'ten 0,0216'ya indirmişti.

**Çizelge 19.** Eşik seçenekleri.

| Eşik | MCC | Recall | Precision | Yanlış alarm |
|---|---|---|---|---|
| 0,50 | 0,6270 | 0,6383 | 0,9357 | 347 |
| 0,77 (MCC enbüyük) | 0,6368 | 0,6085 | 0,9715 | 141 |
| 0,70 | 0,6357 | 0,6165 | 0,9634 | 185 |
| 0,90 | 0,6326 | 0,5885 | 0,9843 | 74 |

## 4.8 Önceki Kurulum: 6 s Penceresi

Aşağıdaki ölçümler 6 s penceresiyle (`[P − 2 s, P + 4 s]`) yapılmıştır.
Bölüm 3.3'te bildirildiği üzere bu pencerelerin %28,8'i S varışını
içermektedir; sonuçlar geçerlidir ancak yalnızca P yapılandırmasından farklı
bir soruyu yanıtlamaktadır.

### 4.8.1 Alan içi başarım

**Çizelge 20.** Yapılandırmalara göre test ROC-AUC değerleri.

| Yapılandırma | Veri kümesi | Tohum başına | Ort. | Std | Topluluk | Taban | Katkı |
|---|---|---|---|---|---|---|---|
| **2B** | **Zor negatif** | 0,9876 / 0,9874 / 0,9864 | 0,9871 | 0,0005 | **0,9882** | 0,9049 | **+0,0833** |
| 2B | Katalog, rastgele gürültü † | 0,9884 / 0,9880 / 0,9878 | 0,9881 | 0,0002 | — | 0,9535 | +0,0350 |
| 2B | Özgün (kapılı) † | 0,9783 / 0,9782 / 0,9773 | 0,9779 | 0,0005 | — | 0,9461 | +0,0318 |
| Kapılı birleştirme | Özgün (kapılı) | 0,9727 / 0,9737 / 0,9733 | 0,9732 | 0,0004 | 0,9745 | 0,9461 | +0,0284 |
| Doğrusal birleştirme | Özgün (kapılı) | 0,9695 / 0,9726 / 0,9734 | 0,9718 | 0,0017 | 0,9730 | 0,9461 | +0,0269 |
| 1B | Özgün, genlik korunmuş | 0,9424 / 0,9438 / 0,9423 | 0,9428 | 0,0007 | 0,9443 | 0,9461 | **−0,0018** |
| 1B | Özgün, pencere bazlı norm. | 0,9139 / 0,9116 / 0,9110 | 0,9122 | 0,0013 | 0,9165 | 0,9205 | −0,0040 |

Katkı = Topluluk − Taban. Bir **Topluluk** sütunu eklenmiştir: önceki sürümde
katkı bazı satırlarda ortalamadan, bazılarında topluluk değerinden
hesaplanmıştır. İki satırda tutarsızlık bulunmuştur. **1B/genlik korunmuş**
satırının katkısı yanlış tabana (`seq` std, 0,9440) göre yazıldığından
**+0,0003** görünmekteydi; doğru değer **−0,0018**'dir ve işareti
değişmektedir: satır tabanla aynı düzeyde değil, tabanın **altındadır**.
**1B/pencere bazlı norm.** satırında ise **−0,0020** yazılıydı; oysa bildirilen
ortalama ile taban arasındaki fark −0,0061'dir. Bu satır yeniden ölçülmüş olup
topluluk değeri 0,9165, katkısı **−0,0040**'tır.

† ile işaretlenmemiş satırlar `--seq-transform asinh` altında yeniden
ölçülmüştür (bkz. Bölüm 3.10). † satırları yeniden ölçülmemiştir; bu iki satır `seq`
kanalını hiç kullanmadığından (2B) Sınırlılık 7 kapsamına girmemektedir. Doğrusal birleştirme satırı bu çalışmada
ilk kez ölçülmüştür.

En iyi tekil kanal yapılandırmasının topluluk ölçütleri (2B, zor negatif,
n = 15.812): accuracy 0,9634 · MCC 0,9279 · PR-AUC 0,9915 · ROC-AUC 0,9882.
Bu kümedeki en iyi genel sonuç birleştirilmiş modele aittir (ROC-AUC 0,9908;
Bölüm 4.8).

Özgün kümede 1B kolu, genlik korunmuş kurulumda tabanın 0,0018, pencere
bazlı normalize kurulumda 0,0040 altındadır. Birleştirme, kapılı yapılandırmada
0,9745, doğrusal yapılandırmada 0,9730 vermekte; tekil 2B kolu 0,9779'dur.
Yeniden kurulan kümelerde tohum std değerleri 0,0004–0,0007, özgün kümede
0,0017–0,0021'dir.

### 4.8.2 Gürültü rejimleri arası aktarım

Her iki küme olaylar, bölümler ve istasyon ataması bakımından özdeş olduğundan
yalnızca zor negatiflerin etkisi yalıtılmaktadır.

**Çizelge 21.** Gürültü rejimleri arası çapraz değerlendirme (ROC-AUC).

| Eğitim ↓ / Değerlendirme → | Rastgele gürültü (taban 0,9535) | Zor negatif (taban 0,9049) |
|---|---|---|
| Rastgele gürültü | 0,9885 (+0,0350) | **0,9841 (+0,0792)** |
| Zor negatif | 0,9873 (+0,0338) | **0,9896 (+0,0847)** |

Yalnızca rastgele gürültüyle eğitilen model, hiç görmediği yüksek genlikli
gürültü geçicilerinde 0,9841 vermektedir; aynı kümede genlik skaleri 0,9049'dur.
Zor negatif kümesi üzerinde iki eğitim rejiminin katkıları +0,0792 ve +0,0847,
aradaki fark 0,0055'tir.

### 4.8.3 Pencere uzunluğu: 3 s ve 6 s

Pencere uzunluğunu yalıtmak için her iki yapılandırma **aynı yordamla** yeniden
üretilmiştir: aynı katalog sabitleme, aynı normalizasyon, aynı zor negatif
madenciliği ve aynı STFT parametreleri (n_fft = 64, hop = 16). Karşılaştırmaya,
3 s pencerelerin 6 s kayıtlardan %50 örtüşmeli **kayan pencere** ile türetildiği
üçüncü bir küme de dâhil edilmiştir; bu kümede varış t = 2,0 s'de bulunduğundan
üç alt pencerenin yalnızca ikisi başlangıcı içermektedir.

**Çizelge 22.** Pencere uzunluğu ve çıkarım yöntemine göre başarım.

| | 3 s kayan | 3 s sabitli | 6 s sabitli |
|---|---|---|---|
| Tohum başına AUC | 0,9086 / 0,9112 / 0,9069 | 0,9792 / 0,9786 / 0,9789 | 0,9876 / 0,9874 / 0,9864 |
| Tohum std | 0,0018 | **0,0002** | 0,0005 |
| **Topluluk ROC-AUC** | 0,9107 | **0,9805** | **0,9882** |
| Koşullu taban | 0,7412 | 0,8481 | 0,9049 |
| Katkı | +0,1695 | +0,1324 | +0,0833 |
| **Kazanılan pay** | **%65,5** | **%87,2** | **%87,6** |
| Accuracy / MCC | 0,8713 / 0,7578 | 0,9537 / 0,9083 | 0,9634 / 0,9279 |

Yeniden üretilen 6 s kümesi, yayımlanan ölçüt kümesinin bölüm boyutlarını
(38.247 / 9.415 / 7.906) ve tabanını (0,9049) birebir yeniden üretmekte; topluluk
AUC değeri 0,9882 olup yayımlanan 0,9892'den 0,0010 farklıdır. İki kurulumun STFT
parametreleri farklıdır (n_fft 256 → 64); aradaki 0,0010'luk fark, aynı
yapılandırmanın tohum yayılımının (0,0005) iki katı mertebesindedir.

Taban 6 s'de 0,9049, 3 s sabitlemede 0,8481, 3 s kayan pencerede
0,7412'dir. Aynı 3 s uzunlukta, kayan pencereden katalog sabitlemeye geçiş
topluluk AUC'yi 0,9107'den 0,9805'e taşımaktadır (+0,0698); aynı geçişte taban
0,7412'den 0,8481'e yükselmektedir. Kazanılan pay 3 s sabitlemede %87,2, 6 s
sabitlemede %87,6'dır.

### 4.8.4 1B kolunun mimarisi

Bölüm 4.8.1'deki 1B sonuçları, 600 ham örneği doğrudan bir BiLSTM'e veren, yani
**hiçbir evrişimli ön uç içermeyen** bir kolla elde edilmiştir. Karşılaştırma
konusu güncel dedektörler bunun tersini yapmaktadır: EQTransformer
CNN → BiLSTM → dikkat sırasını, PhaseNet ise 1B evrişimlerden oluşan bir U-Net'i
kullanmaktadır. Kol bu nedenle zor negatif kümesi üzerinde üç yapılandırmada
yeniden sınanmıştır. Karşılaştırmanın tek değişkenli kalması için spektrogram
kolu tümüyle devre dışı bırakılmış (`--channels 1d`), üç yapılandırma da aynı
tohumlar, aynı bölümleme ve aynı asinh dönüşümü ile çalıştırılmıştır.

**Çizelge 23.** 1B kol mimarisine göre başarım (zor negatif kümesi, taban
0,9049, n = 15.812).

| Yapılandırma | Tohum başına | Ort. | Std | Topluluk | Katkı | Parametre |
|---|---|---|---|---|---|---|
| **Evrişim + BiLSTM** | 0,9891 / 0,9882 / 0,9892 | **0,9888** | 0,0005 | **0,9896** | **+0,0847** | 142.059 |
| Yalnız BiLSTM | 0,9869 / 0,9876 / 0,9874 | 0,9873 | 0,0003 | 0,9883 | +0,0834 | 76.707 |
| Yalnız evrişim | 0,9818 / 0,9826 / 0,9836 | 0,9827 | 0,0007 | 0,9843 | +0,0794 | 48.555 |

**Çizelge 24.** Aynı yapılandırmaların hata bileşimi (eşik 0,5; 7.906 olay ve
7.906 gürültü penceresi).

| Yapılandırma | Kaçırılan olay | Yanlış alarm | Precision | Recall |
|---|---|---|---|---|
| Evrişim + BiLSTM | **410** | 141 | 0,9815 | 0,9481 |
| Yalnız BiLSTM | 546 | 144 | 0,9808 | 0,9309 |
| Yalnız evrişim | 664 | 145 | 0,9804 | 0,9160 |

Yapılandırmalar arasındaki fark, precision'da değil recall'dadır: precision
üçünde de 0,9804–0,9815 aralığındayken kaçırılan olay sayısı 664'ten 410'a
inmektedir. Evrişim + BiLSTM ile yalın BiLSTM arasındaki fark tohum
ortalamalarında 0,0015, toplulukta 0,0013'tür ve tohum aralıkları
örtüşmemektedir (0,9882–0,9892'ye karşılık 0,9869–0,9876);
yapılandırma başına tohum sayısı üçtür. Yalnızca dalga biçimi kullanan
yapılandırma ile 2B kolunun topluluk değerleri 0,9896 ve 0,9882'dir.

### 4.8.5 İki kolun birleştirilmesi

Birleştirme, zor negatif kümesinde iki 1B kolu yapılandırmasıyla ölçülmüştür:
evrişimli ön uçlu ve ön uçsuz. İkinci yapılandırma, birleştirmenin katkısının
güçlü koldan mı yoksa birleştirmenin kendisinden mi geldiğini ayırmak için
çalıştırılmıştır.

**Çizelge 25.** Birleştirilmiş modelin başarımı (doğrusal birleştirme, zor
negatif kümesi, taban 0,9049).

| Yapılandırma | Tohum başına | Ort. | Std | Topluluk | Katkı |
|---|---|---|---|---|---|
| Birleştirme (1B: evrişim + BiLSTM) | 0,9902 / 0,9905 / 0,9897 | 0,9901 | 0,0003 | **0,9908** | **+0,0859** |
| Birleştirme (1B: yalın BiLSTM) | 0,9901 / 0,9904 / 0,9898 | 0,9901 | 0,0002 | 0,9907 | +0,0858 |
| Yalnız 1B (evrişim + BiLSTM) | 0,9891 / 0,9882 / 0,9892 | 0,9888 | 0,0005 | 0,9896 | +0,0847 |
| Yalnız 2B | 0,9876 / 0,9874 / 0,9864 | 0,9871 | 0,0005 | 0,9882 | +0,0833 |

**Çizelge 26.** Hata bileşimi (eşik 0,5; 7.906 olay ve 7.906 gürültü penceresi).

| Yapılandırma | Kaçırılan olay | Yanlış alarm | Precision | Recall |
|---|---|---|---|---|
| Birleştirme (evrişim + BiLSTM) | 429 | 75 | 0,9901 | 0,9457 |
| Birleştirme (yalın BiLSTM) | 449 | **64** | **0,9915** | 0,9432 |
| Yalnız 1B (evrişim + BiLSTM) | 410 | 141 | 0,9815 | 0,9481 |
| Yalnız 2B | 489 | 90 | 0,9880 | 0,9381 |

Birleştirilmiş modelin topluluk AUC değeri 0,9908, tekil kolların
değerleri 0,9896 ve 0,9882'dir; tohum aralıkları örtüşmemektedir
(0,9897–0,9905'e karşılık 0,9882–0,9892). Yanlış alarm sayısı 64–75'tir; tekil
kollarda 141 ve 90'dır. Recall iki kolun arasında kalmaktadır. İki birleştirme
kolunun tohum ortalamaları birbirinin aynıdır (her ikisi de 0,9901) ve topluluk
değerleri arasındaki fark 0,0001'dir.

### 4.8.6 Genlik silindiğinde geriye ne kalıyor?

Bölüm 4.8.1'nin en güçlü olumsuz sonucu, 1B kolunun genliğin ötesinde katkı
sağlamadığıdır. Bu sonuç, evrişimsiz bir kolla ölçülmüştür. Aynı ölçüm burada
Bölüm 4.8.4'nın mimarileriyle yinelenmektedir.

Ölçüm için pencere bazlı normalize küme kullanılmaktadır: standartlaştırma her
pencereyi kendi istatistiklerine göre ölçeklediğinden **mutlak genlik
silinmektedir**. Bunun sonucu tabanlarda doğrudan görülmektedir.

**Çizelge 27.** Pencere bazlı normalize kümede öğrenmesiz tabanlar
(n = 9.548).

| İstatistik | ROC-AUC |
|---|---|
| `seq` std | **0,5000** — tam rastlantı |
| `seq` mutlak maksimum | 0,7088 |
| `img` ortalama dB | **0,9205** — en güçlü önemsiz taban |

`seq` std tabanı tam olarak 0,5000'dir; `seq` mutlak maksimum tabanı
0,7088'e inmektedir. Bu kümede 1B kolunun girdisinde mutlak genlik
bulunmamaktadır.

**Çizelge 28.** 1B mimarilerine göre başarım (pencere bazlı normalize küme).

| Yapılandırma | Tohum başına | Ort. | Std | Topluluk | `seq` tabanına göre | `img` tabanına göre |
|---|---|---|---|---|---|---|
| **Evrişim + BiLSTM** | 0,9301 / 0,9237 / 0,9278 | 0,9272 | 0,0026 | **0,9309** | **+0,2221** | **+0,0104** |
| Yalnız evrişim | 0,9090 / 0,9126 / 0,9084 | 0,9100 | 0,0019 | 0,9146 | +0,2058 | −0,0059 |
| Yalın BiLSTM | 0,9139 / 0,9116 / 0,9110 | 0,9122 | 0,0013 | 0,9165 | +0,2077 | −0,0040 |

Üç yapılandırma da `seq` tabanını (0,7088) 0,20'nin üzerinde geçmektedir.
Yalın BiLSTM (0,9165) ve yalnız evrişim (0,9146) `img` tabanının (0,9205)
altında, evrişim + BiLSTM (0,9309) üzerindedir.

**Genlik korunduğunda aynı mimariler.** Aynı üç mimari, genliğin korunduğu
özgün küme üzerinde de ölçülmüştür (Çizelge 29). Sonuç, yukarıdakinin karşıt
denetimidir.

**Çizelge 29.** Genlik korunmuş özgün kümede 1B mimarileri (taban 0,9461).

| Yapılandırma | Tohum başına | Ort. | Std | Topluluk | Katkı |
|---|---|---|---|---|---|
| Evrişim + BiLSTM | 0,9411 / 0,9434 / 0,9434 | 0,9426 | 0,0011 | 0,9459 | −0,0002 |
| Yalın BiLSTM | 0,9424 / 0,9438 / 0,9423 | 0,9428 | 0,0007 | 0,9443 | −0,0018 |
| Yalnız evrişim | 0,9400 / 0,9433 / 0,9414 | 0,9416 | 0,0014 | 0,9428 | −0,0033 |

Hiçbiri tabanı geçmemekte, tohum ortalamaları birbirinden ayırt
edilememektedir (0,9426 / 0,9428 / 0,9416). Pencere bazlı normalize kümede aynı
mimariler arasındaki fark 0,0165'tir. İki kümenin tabanları sırasıyla 0,9461 ve
0,7088 olup 1,0'e kalan açıklık 0,0539 ve 0,2912'dir.

**Bu kümede birleştirme.** Aynı küme üzerinde 2B kolu ve birleştirilmiş model
de ölçülmüştür.

**Çizelge 30.** Pencere bazlı normalize kümede kanal yapılandırmaları.

| Yapılandırma | Tohum başına | Ort. | Std | Topluluk | `img` tabanına göre |
|---|---|---|---|---|---|
| Yalnız 2B | 0,9770 / 0,9776 / 0,9799 | 0,9782 | 0,0013 | **0,9794** | +0,0589 |
| Birleştirme, doğrusal (1B: evrişim + BiLSTM) | 0,9681 / 0,9646 / 0,9695 | 0,9674 | 0,0021 | 0,9692 | +0,0487 |
| Yalnız 1B (evrişim + BiLSTM) | 0,9301 / 0,9237 / 0,9278 | 0,9272 | 0,0026 | 0,9309 | +0,0104 |

Birleştirme, 1B kolunu tek başına 0,038 geçmekte; ancak 2B kolunun **0,0102
altında** kalmaktadır. Özgün küme üzerinde ölçülen iki birleştirme türüyle
(Çizelge 20) birlikte değerler aşağıdaki gibidir.

**Çizelge 31.** Birleştirilmiş modelin tekil 2B koluna göre farkı, dört veri
kümesi kurulumunda.

| Küme | Birleştirme | 2B | Fark |
|---|---|---|---|
| Özgün, genlik korunmuş | 0,9745 (kapılı) | 0,9779 | −0,0034 |
| Özgün, genlik korunmuş | 0,9730 (doğrusal) | 0,9779 | −0,0049 |
| Özgün, pencere bazlı norm. | 0,9692 (doğrusal) | 0,9794 | −0,0102 |
| **Zor negatif** | **0,9908 (doğrusal)** | **0,9882** | **+0,0026** |

Üç ölçümde birleştirme 2B kolunun altında kalmakta, dördüncüsünde üstüne
çıkmaktadır. Dört ölçümde de aynı kod, aynı mimariler ve aynı tohumlar
kullanılmış; yalnızca negatiflerin seçim yöntemi değişmiştir.

Denetim olarak 2B kolu iki özgün türev küme üzerinde 0,9794 ve 0,9779
vermektedir. `--baseline` seçeneği yalnızca 1B kanalını etkilediğinden iki
kümenin 2B kanalı özdeştir.

Çizelge 20'de bu satırın tabanı olarak kümenin en güçlü öğrenmesiz
istatistiği (`img` ortalama dB, 0,9205) kullanılmıştır. Çizelge 32 ayrıca kolun
kendi kanalının tabanına (`seq` mutlak maksimum, 0,7088) göre değerleri de
vermektedir. İki tabanın seçimi Bölüm 5.1'de tartışılmaktadır.

## 4.9 Korpuslar Arası Genelleme (STEAD)

STEAD değerlendirmesi ilk turda **yalnızca 6 s yapılandırmasıyla** yapılmıştı;
Çizelge 32 o ölçümü vermektedir. **Yalnızca P yapılandırması 27.08.2026'da
karşılıklı olarak çalıştırılmıştır** ve sonuç 4.9.1'de verilmektedir.

**Çizelge 32.** STEAD üzerinde başarım (6 s yapılandırması).

| Eğitim verisi | Değerlendirme | n | AUC | Taban | Katkı |
|---|---|---|---|---|---|
| Kapılı (P içermeyen pencereler) | Eşleşmiş | 27.378 | 0,9818 | 0,9752 | +0,0066 |
| Katalog sabitli | Eşleşmiş | 27.378 | 0,9971 | 0,9752 | +0,0219 |
| Kapılı (P içermeyen pencereler) | Tam aralık | 50.000 | 0,9235 | 0,9531 | −0,0296 |
| Katalog sabitli | Tam aralık | 50.000 | 0,9693 | 0,9531 | +0,0162 |

> **Uyarı.** STEAD gürültüsü genlik ölçeğinde bu korpusun gürültüsünün ~2
> katıdır (medyan `seq` std 0,98'e karşılık 0,47). STEAD içindeki sıralama
> etkilenmediği için ROC-AUC ve PR-AUC aktarılmakta; accuracy, MCC ve Brier
> skoru aktarılmamaktadır.

### 4.9.1 Karşılıklı ölçüm: yalnızca P yapılandırması (27.08.2026)

Bölüm 5.8/5'teki GPD karşılaştırması tek yönlüdür ve açık bir itirazı vardır:
yerel veride yerel eğitilmiş bir modelin kazanması beklenen sonuçtur, kanıt
değildir. İddia ancak aynı model **başkasının korpusunda**, orada hesaplanan bir
tabana karşı ölçüldüğünde simetrik hâle gelmektedir.

STEAD izleri 60 s / 100 Hz olup etiketli `p_arrival_sample` taşıdığından, bu
projenin pencere geometrisi (P'den 200 örnek önce başlayan 340 örnek) doğrudan
kesilebilmektedir. Pencereler `generate-spec-dual-dataset`'in tükettiği
düzende miniSEED olarak yazıldığından süzgeçleme, taban normalizasyonu, zor
negatif madenciliği ve STFT geometrisi **yapı gereği** özdeştir.

**Çizelge 33.** Karşılıklı ölçüm (STEAD ölçüt kümesi, n = 3.070, taban 0,7728).

| model | ROC-AUC | yakalanan açıklık |
|---|---|---|
| GPD `geofon` | **0,9796** | **%91,0** |
| bu çalışma | 0,9207 | %65,1 |

Kendi ölçüt kümemizde sıralama tersidir (bu çalışma %70,9, `geofon` %51,4;
Sınırlılık 5). **Her model kendi korpusunda kazanmaktadır.** Tek yönlü bir
ölçümün genelleme iddiası olarak okunması bu nedenle yanlıştır.

Kolların aktarılabilirliği ayrıca **eşit değildir**: 1B kolu STEAD'e
aktarılabilmekte (%96,8 yakalanan açıklık), 2B kolu taban altında kalmaktadır
(−%51,7). Ayrıntı: `docs/experiment_stead_reciprocal_2026-08-27.md`.
# 5. TARTIŞMA

## 5.1 Karşılaştırma Tabanının Seçimi Sonucu Belirlemektedir

Bu çalışmanın en aktarılabilir bulgusu, bir modelin başarımının değil, o başarımın
**neye göre** ölçüldüğünün belirleyici olduğudur. En iyi yapılandırmanın
0,9908'lik ROC-AUC değeri, çoğunluk sınıfı tabanına göre 0,491'lik bir katkı
gibi görünmektedir; en güçlü koşullu tabana göre ise katkı **0,0859**'dur.
Aradaki fark yaklaşık bir büyüklük mertebesindedir ve modelden değil, veri
kümesinin kuruluş biçiminden kaynaklanmaktadır.

Tabanın ikinci ve daha az belirgin bir işlevi Bölüm 4.8.6'da ortaya çıkmaktadır:
taban yalnızca katkının **büyüklüğünü** değil, bir farkın **ölçülebilir olup
olmadığını** da belirlemektedir. Genlik korunmuş kümede `seq` tabanı 0,9461'dir
ve 1,0'e yalnızca 0,0539 açıklık bırakmaktadır; bu kurulumda üç ayrı 1B
mimarisi arasındaki fark ölçülememektedir (ortalamalar 0,9416–0,9428). Genlik
silindiğinde aynı taban 0,7088'e düşmekte, açıklık 0,2912'ye çıkmakta ve aynı
mimariler arasında 0,0165'lik bir fark görünür hale gelmektedir. **Doyuma yakın
bir taban, gerçek bir farkı yokmuş gibi gösterebilmektedir.** Bölüm 4.8.2'ün
çapraz değerlendirmesi aynı yöne işaret etmektedir: ayırt edici yetenek zor
negatif eğitimi olmaksızın da büyük ölçüde mevcuttur (0,9841), ancak özgün
kümenin tabanı bunu görünür kılamayacak kadar yüksektir. Bir ablasyonun
"etkisiz" çıkması, etkinin yokluğu kadar tabanın bıraktığı payın darlığı
anlamına da gelebilir; bu ayrım yapılmadan olumsuz sonuç bildirilmemelidir.

Bu, tek bir projeye özgü değildir: pozitiflerin varış merkezli, negatiflerin
ayrıca örneklenmiş sakin aralıklardan çekildiği her veri kümesi aynı yapısal
özelliği taşımaktadır. Jover-Alfaro et al. (2026) bağımsız olarak, %97
üzerinde accuracy bildiren bir iş akışının zaman temelli doğrulama altında %24'e
ve konumlar-arası sınamada rastlantı düzeyine düştüğünü göstermiştir. İki
çalışmanın işaret ettiği kusurlar farklıdır — anılan çalışmada veri sızıntısı,
bu raporda tabanın seçimi — ancak sonuç aynıdır: **bildirilen sayı, ölçüm
protokolünün bir işlevidir.** Her iki denetim birlikte uygulanmalıdır;
istasyon-ayrık bölümleme sızıntıyı, koşullu taban abartılı katkı beyanını
engellemektedir. Koşullu tabanın hesaplanma maliyeti eğitim maliyetinin yanında
ihmal edilebilirdir.

**Tabanın kendisi de yanlış ölçülebilir.** Koşullu taban, çoğunluk sınıfı
tabanına göre büyük bir düzeltmedir; ancak taban ROC-AUC ile hesaplandığında
yalnızca monotonik ilişkileri görmektedir. Yalnızca P kümesinin `band`
rejiminde `P(olay | genlik)` U biçimlidir — en sessiz desilin %67'si olaydır —
ve ROC-AUC bu ilişkiyi 0,6447 olarak bildirmektedir. Aynı tekil skaler
üzerinde eğitim bölümünde uyarlanan bir karar ağacı ise 0,7461'e ulaşmaktadır.
Aradaki **0,1015**, modelin görünürdeki payının önemli bir bölümüdür. Bir
tabanın doğru olması yalnızca doğru karşılaştırma nesnesinin seçilmesine değil,
**doğru istatistikle hesaplanmasına** da bağlıdır.

**Değerlendirme kümesinin seçimi de sonucu kaydırmaktadır.** Bölüm 4.4'te aynı
model, aynı pozitiflerle ve yalnızca negatiflerin seçim yöntemi değiştirilerek
0,82 ile 0,91 arasında ROC-AUC vermektedir; kazanılan pay bakımından aralık %12
ile %65 arasındadır. Bu aralık, raporda ölçülen tüm mimari farklarının bir
mertebe üzerindedir.

Bu üç gösterim aynı yöne işaret etmektedir: **bildirilen sayı, ölçüm
protokolünün bir işlevidir.** Bölüm 2.5'teki çizelge aynı örüntünün yazında da
görüldüğünü göstermektedir: negatiflerin seçimini denetleyen çalışmalar recall
bakımından 0,64–0,69, denetlemeyenler 0,94–0,98 aralığındadır.

## 5.2 Dedektör Dalga Biçimi Karakterini Okumaktadır

Bir tabanı geçmek, modelin genlikten başka bir şey kullandığını
**kanıtlamamaktadır**. Taban tekil bir skalerin ROC-AUC değeridir ve yalnızca
sıralamayı ölçmektedir; aynı skalerin daha iyi biçimlendirilmiş bir işlevi —
doğru yere konmuş bir eşik ya da monotonik olmayan bir tepki — tabanı geçmekle
birlikte özünde bir genlik dedektörü olmayı sürdürebilir. Bu ayrımın bu veri
kümesinde 0,1015 AUC ettiği Bölüm 4.1'de ölçülmüştür.

Bölüm 4.5 genliği sabit tutarak ayrımı yapmaktadır. Genliğin en çok 2,5 kat
değiştiği yedi desilde model 0,63 ile 0,97 arasında ROC-AUC almaktadır.
**Genliğin hiçbir işlevi bu değerleri üretemez**; geriye kalan tek bilgi
kaynağı dalga biçiminin kendisidir. Dedektörün dalga biçimi karakterini okuduğu
bu ölçümle gösterilmektedir.

Desil içi ROC-AUC genlikle birlikte tekdüze artmaktadır (0,63'ten 0,97'ye).
Bu, Bölüm 4.6'daki işletim zarfının SNR bağımlılığını bütünüyle bağımsız bir
hesaptan doğrulamaktadır: dalga biçimi, P varışı gürültünün üzerine çıktıkça
okunabilir hâle gelmektedir.

Ölçümün **desteklemediği** iki nokta ayrıca kaydedilmelidir. En sessiz desil
508 kat genişliktedir; oradaki düşük değer ne bir yetersizliği ne de bir
yeterliliği göstermektedir. İkincisi, `natural` eğitimli modelin dalga biçimini
daha iyi okuduğu söylenememektedir: aynı test kümesinde dar desil medyanları
0,7167 ve 0,7175'tir. Bu modelin üstünlüğü tanımada değil, **yanlış alarmdadır**
(157'ye karşılık 317, eşit recall'da).

## 5.3 Eğitim Dağılımı ile Değerlendirme Dağılımı Ayrı Kararlardır

Genlik eşleştirilmiş küme bir **ölçüm aracıdır**, bir eğitim reçetesi değildir.
Amacı, genliği bilgisiz kılarak geriye kalanı ölçülebilir kılmaktır.
Konuşlandırılmış bir istasyon ise gürültü dağılımının tümünü görmektedir.

Bölüm 4.4 aktarımın **bakışımsız** olduğunu göstermektedir: `natural` eğitimli
model `matched` kümesinde %46,2, `matched` eğitimli model `natural` kümesinde
%13,6 pay kazanmaktadır. Geniş dağılımda eğitilen model her iki yöne de
aktarılabilmekte, dar dağılımda eğitilen model aktarılamamaktadır. Bu, eğitim
kümesinin gerçekçi dağılımdan seçilmesi gerektiğini göstermektedir.

Ancak eğitim dağılımı boşluğun tamamını kapatmamaktadır. `natural` kümesinde
`natural` eğitimi kazanılan payı %13,6'dan %20,7'ye çıkarmakta, yani `matched`
kümesindeki %61,1 ile arasındaki 47,5 puanlık boşluğun yalnızca 7,1 puanını
kapatmaktadır. Kalan bölüm bir eğitim uyumsuzluğu değildir: doğal gürültü
rejiminde genlik skalerinin ötesinde çıkarılabilecek bilgi daha azdır.

## 5.4 Mimari Katkısı Küçüktür — Ancak Sıfır Değildir ve Nerede Aranacağı Belirlenmiştir

Bu çalışmanın ilk turunda sınanan yapısal eklentiler — çift kanallı mimari,
kapılı birleştirme, geç birleştirmeli yığınlama — yalın spektrogram CNN'ini
geçememişti. Bölüm 4.8.4'daki ölçüm bu sonucu bütünüyle geçersiz kılmamakta,
ancak önemli ölçüde niteliklendirmektedir: eklentinin **nereye** yapıldığı
belirleyicidir.

1B kolu içinde mimari fark ölçülebilirdir. Yinelemesiz bir evrişim yığını
0,9843'te kalırken, yineleme içeren yapılandırmalar 0,9883 ve 0,9896'ya
ulaşmaktadır; aradaki ~0,005'lik fark tohum yayılımının yaklaşık on katıdır.
Bu farkın tümü **recall'dan** kaynaklanmaktadır (kaçırılan olay 664 → 410,
precision değişmeksizin). Dolayısıyla "mimari önemsizdir" biçimindeki bir
genelleme desteklenmemektedir; desteklenen ifade, mimari kazanımlarının
0,005 mertebesinde kaldığı, veri kümesi kurulumundan kaynaklanan kazanımların
ise (Bölüm 5.5) bir mertebe büyük olduğudur.

1B kolunun katkısının genlikle sınırlı olduğu yönündeki daha güçlü sonuç ise
Bölüm 4.8.6'da **doğrudan sınanmış ve geçerliliğini korumamıştır**. Genlik
tümüyle silindiğinde (`seq` std tabanı tam 0,5000) evrişimli ön uçlu yinelemeli
kol 0,9309'a ulaşmakta ve hiç görmediği spektrogram tabanını dahi
geçmektedir; yeniden öğrenilecek bir skaler yokken bu başarım ancak dalga
biçimi karakterinden gelebilir. Söz konusu olan, kolun ulaşamadığı bir bilgi
değil, evrişimsiz bir yinelemenin erişemediği bir gösterimdir.

Birleştirme konusundaki daha önceki sonuç da (Çizelge 20) benzer
biçimde niteliklendirilmelidir. Zor negatif kümesinde doğrusal birleştirme her
iki kolu da geçmekte (0,9908; kollar 0,9896 ve 0,9882) ve yanlış alarm sayısını
her iki kolun da altına indirmektedir (64–75; kollar 141 ve 90). Bölüm 4.8.1'de
ölçülen kötüleşme özgün küme üzerinde ve **yalnızca kapılı** birleştirmeyle
elde edilmiştir; dolayısıyla "birleştirme zarar vermektedir" ile "kapı zarar
vermektedir" ayrımı o ölçümle yapılamamaktadır. Bu ayrım özgün küme üzerinde ölçülmüştür (Çizelge 20): kapılı birleştirme
0,9745, doğrusal birleştirme 0,9730 vermekte, ikisi de tekil 2B kolunun
0,9779 değerinin altında kalmaktadır. Kötüleşme kapıya özgü değildir.

Buna karşılık mimari katkının **nerede** ölçüldüğü belirleyicidir: 1B kolu tek
başına çalışırken evrişimli ön uç 0,0015 kazandırırken, spektrogram kolu
devredeyken bu kazanç ölçülemez düzeye inmektedir (Çizelge 25: 0,9908'e karşılık
0,9907). Tek
kollu ablasyonlarda anlamlı görünen bir mimari farkın tam modelde
kaybolabileceği, bu çalışmanın yöntemsel çıkarımlarından biridir.

Eğitim örneği başına ~1,5 parametre ile model kapasite sınırlı değildir; bu
gözlem geçerliliğini korumaktadır.

## 5.5 Veri Kümesi Kurulumu Mimariden Daha Belirleyicidir

STA/LTA tabanlı sabitlemenin P varışını içermediğinin saptanması yalnızca bir
hata düzeltmesi değildir: katalog tabanlı sabitlemeye geçiş kayıt tutulma oranını
%62,6'dan %96,3'e çıkarmış ve tek başına STEAD üzerindeki korpuslar arası katkıyı
**üç katına** çıkarmıştır (+0,0066 → +0,0219). Tam aralıkta taban altı bir sonuç
(−0,0296) taban üstü bir sonuca (+0,0162) dönüşmüştür.

Aynı sonuç pencere geometrisi için de geçerlidir: 3 s yapılandırmasında
pencerelerin kayan pencere yerine katalog sabitlemeyle çıkarılması topluluk AUC'yi
**0,0698** artırmıştır. Bu, projede ölçülen tek en büyük etkidir ve mimariden
değil pencerenin varışa göre nerede kesildiğinden kaynaklanmaktadır;
karşılaştırma olarak tüm mimari eklentilerinin etkisi 0,005 mertebesinde
kalmıştır.

Ayrıca bir seçim yanlılığı giderilmiştir: tetiklenmeyen kaydın elenmesi, pozitif
sınıfı klasik bir dedektörün zaten bulduğu alt kümeye indirgemekte ve elenen
kayıtlar orantısız biçimde düşük SNR'li, yani öğrenen bir dedektörün değerini
gösterebileceği kayıtlar olmaktadır.

**Dört bağımsız ölçüm aynı sonuca çıkmaktadır.** Bunlar farklı deneylerden
gelmekte ve aynı yönü göstermektedir:

| Değişen | Etki |
|---|---|
| Pencere geometrisi (kayan pencere → katalog sabitleme, 3 s) | **0,0698** |
| Sabitleme yöntemi (STA/LTA → katalog), STEAD katkısı | +0,0066 → **+0,0219** |
| Negatif seçimi — birleştirmenin **işareti** | −0,0102 … −0,0034 → **+0,0026** |
| Taban doyumu — 1B mimarileri arasındaki ölçülebilir fark | ~0 → **0,0165** |
| *Karşılaştırma:* tüm mimari eklentileri | 0,0015 – 0,0053 |

![Sekil 2](/home/hogib/Desktop/rapor_sekiller/sekil3_birlestirme_isaret.png)

**Şekil 2.** Birleştirilmiş modelin tekil 2B koluna göre farkı, dört veri kümesi kurulumunda. Aynı kod ve aynı tohumlarla, fark yalnızca büyüklük değil **işaret** değiştirmektedir.

Üçüncü satır özellikle belirleyicidir: aynı kod, aynı tohumlar ve aynı iki
mimariyle, birleştirme üç kümede 2B kolunun altında kalırken zor negatif
kümesinde üstüne çıkmaktadır. Bir mimari kararın yararlı mı zararlı mı olduğu
sorusunun yanıtı, negatiflerin nasıl seçildiğine göre **işaret
değiştirmektedir**. Mimari kazanımları 0,005 mertebesindeyken kurulum
kaynaklı etkiler bir mertebe büyüktür.

## 5.6 İşletim Sınırını Büyüklük Değil SNR Belirlemektedir

Bölüm 4.6'deki ölçüm, dedektörün neyi bulup neyi kaçırdığını tek bir AUC
değerinin gösteremeyeceği biçimde ortaya koymaktadır. Kaçırılan olaylarla
bulunanlar arasındaki medyan fark log SNR'de 1,44 iken büyüklükte 0,2 birim,
uzaklıkta 4,6 km'dir. Büyüklüğe göre görülen eğilim (M 1,5–2,0 için 0,9100'den
M 3,0–3,5 için 0,9900'e) büyük ölçüde SNR'nin dolaylı yansımasıdır; aynı
istasyonda büyük olaylar daha yüksek genlikle kaydedilmektedir.

Bu, 0,9049'luk genlik tabanının doğrudan karşılığıdır: sınırı olayın büyüklüğü
değil, sinyalin yerel gürültünün ne kadar üzerinde olduğu belirlemektedir.
Uygulamaya yönelik sonucu, konuşlandırma ifadesinin büyüklüğe değil SNR'ye
koşullanması gerektiğidir. Aynı sınır büyüklük kestirimi zincirine de
geçmektedir: kaçırılan bir olay hiçbir zaman büyüklük alamamaktadır,
dolayısıyla dedektörün zarfı zincirin de zarfıdır.

**Yalnızca P yapılandırması bu sonucu güçlendirmektedir.** 6 s
yapılandırmasında recall uzaklıkla değişmekteydi (0,9773 → 0,9388). Yalnızca P
yapılandırmasında aynı dilimler 0,6616 / 0,6316 / 0,6334, yani **düz**dür ve
kaçırılan ile bulunan olaylar arasındaki medyan uzaklık farkı 4,59 km'den
0,56 km'ye inmektedir. SNR bağımlılığı ise korunmaktadır (0,469 → 0,945).

S varlığı uzaklıkla neredeyse birebir örtüştüğünden (25 km içinde %99,3, 50 km
ötesinde %0), 6 s yapılandırmasındaki uzaklık etkisinin önemli bir bölümünün S
katkısı olduğu değerlendirilmektedir. Bu, uzaklık ve SNR'nin o kurulumda
ayrıştırılamamasının doğrudan sonucudur.

## 5.7 Konuşlandırılabilir İddia ve Eşik

Bildirilebilecek iddia ROC-AUC değil **recall**'dır: M ≥ 2,0 olaylar için,
56 km yarıçap içinde, P varışından sonraki 1,4 s ile **recall 0,61–0,64**.
Eşik ayarı bunu kurtarmamaktadır (Çizelge 27): MCC'yi enbüyükleyen 0,77
eşiğinde recall 0,6085'tir.

Kalibrasyon, modelin **fazla değil eksik güvenli** olduğunu göstermektedir
(uyarlanan sıcaklık 0,476). Sinir ağlarında sık görülen yön bunun tersidir;
buradaki yönün topluluk ortalamasının logit uzayında alınmasıyla ilişkili
olması olasıdır. Ölçekleme sonrasında ECE dörtte bire inmekte, olasılıklar
karar verilebilir hale gelmektedir.

Eşik seçimi ise tek bir ölçütle kapatılamamaktadır. MCC'yi enbüyükleyen eşik
0,28 olmakla birlikte, 0,50 ile arasındaki fark (0,9402'ye karşılık 0,9391)
tohum yayılımı mertebesindedir; MCC bakımından iki nokta ayırt edilemezdir.
Ayrım, iki hata türünün maliyetinin eşit olmamasından doğmaktadır: 0,28
eşiğinde 45 olay daha bulunmakta, karşılığında 33 yanlış alarm daha
üretilmektedir. Kaçırılan bir olay büyüklük kestirimi zincirine hiçbir zaman
ulaşamamakta, yanlış alarm ise yalnızca ikinci aşamada hesaplama maliyeti
doğurmaktadır. Bu asimetri göz önüne alındığında zincir için düşük eşik
savunulabilirdir; tek başına çalışan bir dedektör için ise yüksek eşik (0,90'da
25 yanlış alarm) tercih edilebilir. Çizelge 25 bu nedenle tek bir çalışma
noktası önermek yerine seçenekleri vermektedir.

**Kalibrasyon bu yapılandırmada başarımı kötüleştirmektedir.** ECE 0,0484'ten
0,0765'e, Brier 0,1386'dan 0,1444'e yükselmektedir; yalnızca MCE
düşmektedir. 6 s yapılandırmasındaki bulgunun tersidir ve bu yapılandırma
**kalibrasyonsuz** bildirilmelidir. Sıcaklık ölçekleme, olasılık kütlesi zaten
uçlarda toplanmış bir modelde işe yaramakta; kütlesi ortada dağılmış bir
modelde uçlara iterek aşırı güven üretmektedir.

Karar gecikmesi tasarım gereği P varışından **1,4 s** sonradır. Bu, yazınla
doğrudan karşılaştırılabilen tek büyüklüktür: NZ edge CNN 2 s, TransQuake
50 s pencere kullanmaktadır.
## 5.8 Sınırlılıklar

1. **Pozitif sınıfta artık etiket gürültüsü.** Katalog, bir depremin
   gerçekleştiğini bildirmektedir; *bu istasyonun* onu kaydettiğini değil.
   Ölçümler pozitiflerin ~%10–15'inin sınırda olduğunu göstermektedir;
   ulaşılabilir tavan 1,0'in altındadır.
2. **Varış doğruluğu yalnızca tespit içindir** (0,63 s medyan mutlak sapma).
3. **Zor negatif kümeleri bilinçli olarak temsili değildir**; kalibre edilmiş
   çalışma noktası değerleri rastgele örneklenmiş test kümesinden alınmalıdır.
4. **Uzaklık aralığı dardır.** İndirme yarıçapı 0,5° olduğundan episantr
   uzaklığı ~56 km ile sınırlıdır (Çizelge 2: p95 53,5 km, maks 55,6 km) ve
   dedektörleri en çok ayrıştıran düşük SNR'li uzak rejim bu aralığın dışında
   kalmaktadır. Bölüm 4.6'de uzaklığın recall üzerindeki zayıf etkisi
   (0,9773 → 0,9388) bu aralık içinde geçerlidir; genel bir sönümlenme ifadesi
   değildir.
5. **Güncel dedektörlerle karşılaştırma yapılmamıştır.** Bölüm 2.5 protokol
   düzeyinde bir karşılaştırma vermektedir; aynı ölçüt kümesi üzerinde
   çalıştırılmış bir karşılaştırma yoktur. *(Eski madde:)* EQTransformer
   (SeisBench, `instance` ağırlıkları) eşleşmiş STEAD kümesinde iki uç
   yapılandırmada çalıştırılmıştır; elde edilen aralık (0,9565–0,9989) bu
   projenin değerini içine almakta, dolayısıyla ölçüm bir sıralama
   vermemektedir (Çizelge 11). İki modelin girdi uzunlukları ve tasarım
   amaçları farklı olduğundan aradaki farkın ne kadarının mimariden
   geldiği bu kurulumla ayrılamamaktadır.

   **Güncelleme (27.08.2026): bu sınırlılık büyük ölçüde giderilmiştir.** GPD
   (Ross vd. 2018) beş ağırlık kümesiyle **bu projenin kendi ölçüt kümesi**
   üzerinde çalıştırılmıştır (14.821 pencere, koşullu taban 0,5860):

   | model | ROC-AUC | yakalanan açıklık |
   |---|---|---|
   | **bu çalışma (birleştirme, 3 tohum)** | **0,8796** | **%70,9** |
   | GPD `geofon` | 0,7987 | %51,4 |
   | GPD `original` | 0,7710 | %44,7 |
   | GPD `stead` | 0,7499 | %39,6 |
   | GPD `instance` | 0,7457 | %38,5 |
   | GPD `scedc` | 0,7154 | %31,2 |

   Karşılıklı ölçüm de yapılmıştır: STEAD'in kendi ölçüt kümesinde (3.070
   pencere, taban 0,7728) sıralama tersine dönmekte — `geofon` 0,9796 (%91,0),
   bu çalışma 0,9207 (%65,1). **Her model kendi korpusunda kazanmaktadır**; bu,
   tek yönlü bir ölçümün genelleme iddiası olarak okunmasının neden yanlış
   olduğunu doğrudan göstermektedir. Ayrıntı: `docs/experiment_gpd_baseline_2026-08-27.md`
   ve `docs/experiment_stead_reciprocal_2026-08-27.md`. PhaseNet hâlâ
   çalıştırılmamıştır.
6. **Tek bölge, tek katalog** ile eğitim yapılmıştır. Sonuçların başka bir
   sismotektonik ortama ve başka bir katalog uygulamasına aktarılabilirliği
   ölçülmemiştir; Bölüm 4.9'teki STEAD değerlendirmesi bu soruyu yalnızca
   kısmen yanıtlamaktadır. Olasılıklar Bölüm 4.7'da sıcaklık ölçeklemeyle
   kalibre edilmiştir; ölçekleme parametresi bu korpusa özgüdür ve başka bir
   korpusa aktarılmamalıdır.
7. **Katalog eksikliği saptanmış ve giderilmiştir (30.08.2026).** Bu çalışmada
   30.08.2026'ya kadar kullanılan katalog kopyası, bölgeye ait AFAD olaylarının
   ~%29'unu içermemekteydi; eksiklik zamansal değil **uzamsaldır** (dosya
   12.08.2026'ya kadar günceldi, ancak Şubat 2025'te bölgedeki 1.256 olaydan
   yalnızca 51'ini tutuyordu). Katalog AFAD API'sinden yeniden kurulmuştur
   (`scripts/fetch_afad_catalog.py`; tek istek, <30 s). **Tespit sonuçları
   etkilenmemektedir** (Bölüm 3.1'deki denetim: 55.595'te 3 pencere). Tahmin
   sonuçları etkilenmektedir ve Bölüm 6'da verilmiştir.

8. **Mixed precision'da sayısal taşma riski (Bölüm 3.10) taranmış ve
   giderilmiştir.** Risk yalnızca `seq` kanalını kullanan yapılandırmaları
   ilgilendirmektedir. Kümeler tek tek taranmıştır: özgün genlik korunmuş küme
   71.672 pencerenin 4'ünde fp16 sınırını aşmakta, pencere bazlı normalize
   küme ise hiç aşmamaktadır (en büyük değer 21,6). Etkilenebilecek iki satır
   (1B/genlik korunmuş ve kapılı birleştirme) asinh altında yeniden ölçülmüş ve
   **özgün tohum yayılımları içinde yeniden üretilmiştir**; dolayısıyla hiçbir
   sonuç taşmadan etkilenmemiştir. 3 s kümesi de taşmaktadır (en büyük değer
   1,21 × 10⁶), ancak o kümede bildirilen sonuçlar yalnızca 2B kanalıyla
   üretildiğinden etkilenmemektedir; bu kümede ileride yapılacak 1B veya
   birleştirme ölçümleri asinh gerektirmektedir.
8. **Recall ölçümleri dengeli ölçüt kümesine aittir.** Sürekli veride taban
   oranı bambaşkadır. Dengeli test kümesinde ölçülen yanlış alarm oranı
   birleştirilmiş modelde %0,95 (75/7.906), yalnız 1B kolunda %1,78'dir
   (141/7.906); 6 s pencere ve 1 s adımla sürekli veriye taşındığında bu
   oranlar istasyon başına günde sırasıyla ~820 ve ~1.540 yanlış alarma
   karşılık gelmektedir. Recall aktarılabilir, precision aktarılamaz.

9. **Yalnızca P güvencesi hız modeline görelidir.** En yakın varışın kesme
   noktasına payı 0,050 s'dir; S−P kestiriminde 0,63 s'lik bir sapma 2.021
   kaydı (%3,6) riske sokmaktadır (Çizelge 5).
10. **Genlik eşleştirmesi havuz derinliğiyle sınırlıdır.** Olay `seq` std
    değeri 5.439'a ulaşırken havuzdaki en yüksek gürültü penceresi 77,8'dir.
    Eşleştirme dağılımın alt yarısında tutmakta (p1–p50 arasında 1,15–1,35
    kat), üst çeyrekte bozulmaktadır (p99'da 14,9 kat). Kalan ayrım gücü bu
    eşleştirilemeyen kuyruktan gelmektedir; bu kuyruk gerçek bir fiziksel
    ayrımdır, madenciliğin ürünü değildir.
11. **Sürekli veride yanlış alarm oranı ölçülmemiştir.** Yazındaki
    konuşlandırılmış her sistemin bildirdiği büyüklük budur. BODT istasyonu
    tespit korpusunun eğitim bölümünde bulunduğundan bu ölçüm için
    kullanılamaz; test bölümünde 35 istasyon ayrılmıştır.
12. **Korpuslar arası sınama yalnızca 6 s yapılandırması için yapılmıştır**
    (Bölüm 4.9).
13. **2B kolunun zaman çözünürlüğü sınırlıdır.** `hop = 16` ile her kare
    160 ms'dir; P başlangıcı onlarca ms içinde gerçekleşmektedir. Bu
    parametre ölçülerek değil benzeşimle seçilmiştir.

## 5.9 Sonraki Adımlar

Öncelik sırasıyla:

*(31.08.2026'da güncellenmiştir: 1. ve 3. maddeler tamamlanmıştır.)*

1. ~~**Yalnızca P yapılandırmasının STEAD üzerinde korpuslar arası
   sınanması.**~~ **Tamamlandı (27.08.2026).** Karşılıklı olarak yapılmıştır ve
   sonuç her iki yönde de ölçülmüştür: her model kendi korpusunda kazanmaktadır
   (Sınırlılık 5). Ayrıca 1B kolu STEAD'e aktarılabilirken (%96,8 yakalanan
   açıklık) 2B kolu taban altında kalmaktadır (−%51,7) — kolların
   aktarılabilirliği eşit değildir.
2. ~~**PhaseNet ve GPD ile bu korpusun kendi ölçüt kümesinde
   karşılaştırma.**~~ **GPD için tamamlandı (27.08.2026)**, beş ağırlık kümesiyle
   (Sınırlılık 5). **PhaseNet hâlâ açıktır.**
3. **Sürekli veride yanlış alarm oranının doğrudan ölçülmesi**, test
   bölümündeki 35 istasyondan birinden çekilecek sürekli kayıtla. Bu, yazınla
   karşılaştırmayı mümkün kılan eksik büyüklüktür. **Şu an en yüksek öncelikli
   açık madde.**
4. **Ağ eşzamanlılığı (2-of-N istasyon) ile yanlış alarm bastırma.**
   Bağımsız yanlış pozitifler %1,78'den 2-of-2'de ~%0,03'e inmektedir — sürekli
   konuşlandırmanın ihtiyaç duyduğu büyüklük mertebesi budur. **Sınırlılığı
   önden belirtmek gerekir:** test kümesindeki 6.459 olayın yalnızca 1.184'ünde
   ikinci bir istasyon mevcuttur, dolayısıyla ölçüm bir alt küme üzerindedir.
5. **2B kolunun STFT zaman çözünürlüğünün taranması** (Sınırlılık 13).
6. **İndirme yarıçapının genişletilerek düşük SNR'li uzak rejimin
   kapsanması.** Bölüm 4.6 işletim sınırını SNR'nin belirlediğini
   gösterdiğinden, ayırt ediciliğin en çok sınanacağı aralık bugün veri
   kümesinde bulunmamaktadır.
7. **CNN-GRU kolu** (`--branch-1d cnn-gru`): 4.8.4'teki ızgara yinelemenin
   taşıyıcı olduğunu gösterdiğinden (0,9896'ya karşı 0,9843, tohum aralıkları
   örtüşmüyor) ucuz ve gerekçeli bir dördüncü koldur.


# 6. TAHMİN: PROJENİN ÖZGÜN SORUSUNA SINIRLANDIRILMIŞ YANIT

*(Bu bölüm 31.08.2026'da eklenmiştir. Raporun 2–5. bölümleri tespit ve büyüklük
işine aittir; projenin özgün hedefi ise **katalog verisinden olay zamanı ve
sınıfı tahmini**dir. Aşağıdaki sonuç o soruyu iki yandan sınırlandırmaktadır.)*

## 6.1 Sonuç: sinyal katalogda, sismogramda değil

Tahmin özelliklerinin kaynağına göre sonuç kesin biçimde ayrışmaktadır.

**Katalog türevli özellikler kalıcılık tabanını geçmektedir.** Fay bölgesi
bazında, 30 günlük ayrık bloklarda (dürüst örneklem büyüklüğü; ardışık
pencereler 11–46 kat örtüştüğünden pencere düzeyinde havuzlama AUC'yi +0,25 ile
+0,35 arasında şişirmektedir):

| bölge | blok | temel oran | eski katalog | **düzeltilmiş** | fark |
|---|---|---|---|---|---|
| **EGE** | 43 | 0,581 | 0,5190 ±0,0150 | **0,6918 ±0,0165** | **+0,173** |
| **ORTA** | 43 | 0,395 | 0,3960 ±0,0335 | **0,6176 ±0,0346** | **+0,222** |
| DAFZ | 47 | 0,596 | 0,6615 ±0,0173 | 0,6667 ±0,0323 | +0,005 |
| KAFZ | 42 | 0,381 | 0,4643 ±0,0346 | 0,4103 ±0,0011 | −0,054 |

**Dalga biçimi türevli özellikler geçmemektedir.** Değerlendirmenin geçerli
olduğu bir çalışma noktasında (M≥4,0, 14 gün; özgün M≥4,5/30 gün kurulumunda
katlar dejenere olmakta, iki AUC tanımsız kalmaktadır), üç dizi mimarisi de
0,5823'lük kalıcılık tabanının altındadır: LSTM 0,5244, GRU 0,5709, TCN 0,5204.
Kaotik özellikler (permütasyon entropisi, Hjorth, örneklem entropisi vb.) aynı
sonucu vermektedir: dört model türevinin tamamında taban altında, 10 bağlam ×
ufuk hücresinin 0'ında üstünde.

**Bu iki ölçüm birlikte projenin özgün sorusunu sınırlandırmaktadır: bu
kurulumda tahmin sinyali deprem katalogundan gelmekte, sismogramdan
gelmemektedir.**

## 6.2 Katalog düzeltmesinin sonuçları iki yöne birden taşıması

Bölüm 5.8/7'de anlatılan katalog eksikliği giderildiğinde sonuçlar **iki yöne
birden** hareket etmiştir; bu, ölçümün ayarlama değil ölçüm olduğunun
göstergesidir:

- **Kaotik özellik sonuçları kötüleşmiştir.** Eksik olaylar geri geldiğinde
  BODT'de 6 saatlik ufukta pozitif oran %25,1'den %39,9'a çıkmakta, kalıcılık
  tabanı 0,5423'ten 0,5713'e yükselmektedir. Yoğun bir artçı dizisi tam olarak
  "önceki olaydan bu yana geçen gün" değişkeninin iyi kestirdiği şeydir;
  dalga biçimi özellikleri bu yükselişe yetişememektedir.
- **Katalog türevli model sonuçları iyileşmiştir.** Bu model özelliklerini de
  katalogdan üretmektedir (olay sayıları, b-değeri, büyüklük açığı), dolayısıyla
  kataloğun tamamlanması gördüğü girdiyi iyileştirmektedir.

Kazanımların **uzamsal dağılımı** bunun varyans artefaktı olmadığının en güçlü
kanıtıdır: eksik olaylar ezici çoğunlukla Ege açıklarındaydı ve iyileşme EGE
(+0,173) ile ona komşu ORTA (+0,222) bölgelerinde, tohum yayılımının beş–on
katı büyüklükte gerçekleşmektedir; doğuda kalan DAFZ değişmemekte (+0,005),
KAFZ rastlantı düzeyinde kalmaktadır. Bir varyans artefaktı bu coğrafyaya uymaz.

**ORTA bölgenin rastlantıdan 0,618'e çıkması bu projenin tahmin sonuçlarındaki
en büyük tekil değişimdir.** Önceki raporun bu bölgeyi "Poisson'a yakın (CV ≈ 1),
dolayısıyla tahmin edilemez" biçimindeki fiziksel teşhisi düzeltilmiş katalogla
geçerliliğini korumamaktadır. Aynı teşhis KAFZ için geçerliliğini korumaktadır.

## 6.3 Yöntemsel uyarı: sessiz atlama

Bu bölümün hazırlanmasında üç ayrı hata, hata mesajı üretmeden "başarı" olarak
raporlanmıştır: (i) bir koşum betiği `[ERROR]` yazdırıp 0 ile çıkan bir süreci
başarılı saymıştır; (ii) özellik yükleyicisi `Zaman_Dk` alanını yanlış
yorumlayarak 1.238.672 pencereyi tek bir 1970 saatine indirmiş, arşiv **2 saatlik
öznitelik vektörü** olarak yüklenmiştir; (iii) blok düzeyi değerlendirme, bir
yol argümanı verilmediği için sessizce atlanmış, yalnızca şişirilmiş pencere
düzeyi AUC basılmıştır (0,8558'e karşı gerçek 0,5102). **Eksik bir sayı,
önemsiz bir sayı değildir**; bu kod tabanı gürültülü değil sessiz biçimde
bozulmaktadır.

Ayrıntı: `docs/experiment_neural_forecasters_2026-08-30.md` ve
`docs/experiment_chaos_forecast_2026-08-27.md`.

# KAYNAKLAR (taslak — künyeler kendi .bib dosyanızla doğrulanmalıdır)

- Albelali, S. & Ahmed, M. (2025). Hidden leaks in time series forecasting: How
  data leakage affects LSTM evaluation across configurations and validation
  strategies. *arXiv:2512.06932*.
- Allen, R. V. (1978). Automatic earthquake recognition and timing from single
  traces. *Bulletin of the Seismological Society of America*, 68(5), 1521–1532.
- Başar, D. & Çelik, R. N. (2026). A hybrid CNN-LSTM architecture for seismic
  event detection using high-rate GNSS velocity time series. *Sensors*, 26(1),
  519.
- Crotwell, H. P., Owens, T. J. & Ritsema, J. (1999). The TauP Toolkit: Flexible
  seismic travel-time and ray-path utilities. *Seismological Research Letters*,
  70(2), 154–160.
- Jover-Alfaro, J., Arias-Antúnez, E. & Mateo-Cortés, J. A. (2026). Forecasting
  earthquakes by Machine Learning techniques: lights and shadows. *Earth Science
  Informatics*, 19, 24.
- Kennett, B. L. N. & Engdahl, E. R. (1991). Traveltimes for global earthquake
  location and phase identification. *Geophysical Journal International*,
  105(2), 429–465.
- Mousavi, S. M., Sheng, Y., Zhu, W. & Beroza, G. C. (2019). STanford EArthquake
  Dataset (STEAD): A global data set of seismic signals for AI. *IEEE Access*,
  7, 179464–179476.
- Mousavi, S. M., Ellsworth, W. L., Zhu, W., Chuang, L. Y. & Beroza, G. C.
  (2020). Earthquake transformer — an attentive deep-learning model for
  simultaneous earthquake detection and phase picking. *Nature Communications*,
  11, 3952.
- Perol, T., Gharbi, M. & Denolle, M. (2018). Convolutional neural network for
  earthquake detection and location. *Science Advances*, 4(2), e1700578.
- Ross, Z. E., Meier, M.-A., Hauksson, E. & Heaton, T. H. (2018). Generalized
  seismic phase detection with deep learning. *Bulletin of the Seismological
  Society of America*, 108(5A), 2894–2901.
- Stockman, S., Lawson, D. & Werner, M. (2026). EarthquakeNPP: A benchmark for
  earthquake forecasting with neural point processes. *Transactions on Machine
  Learning Research*.
- Wang, L. & Zhao, W. (2025). An ensemble deep learning network based on 2D
  convolutional neural network and 1D LSTM with self-attention for bearing fault
  diagnosis. *Applied Soft Computing*, 172, 112889.
- Woollam, J. et al. (2022). SeisBench — A toolbox for machine learning in
  seismology. *Seismological Research Letters*, 93(3), 1695–1709.
- Zhu, W. & Beroza, G. C. (2019). PhaseNet: A deep-neural-network-based seismic
  arrival-time picking method. *Geophysical Journal International*, 216(1),
  261–273.