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
Bölüm 3.8).

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
bölümleme** (Bölüm 3.4), **koşullu genlik tabanı** (Bölüm 3.8) ve yeniden eğitim
yapılmaksızın **korpuslar arası sınama** (Bölüm 4.4).

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

# 3. GEREÇ VE YÖNTEM

## 3.1 Veri

Olaylar, Kandilli Rasathanesi ve Deprem Araştırma Enstitüsü (KRDAE/KOERI)
bölgesel kataloğundan alınmıştır. İki katalog dosyası farklı amaçlarla
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

**Çizelge 1.** Korpusa giren olayların özellikleri.

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
denetlenerek ±300 s içinde katalog kaydı bulunanlar elenmiştir. Denetim yalnızca
zamansaldır; bu, bilinçli olarak aşırı temkinli bir tercihtir. Kullanılabilir
gürültü sinyalden yaklaşık 50 kat fazladır (1.784.650'ye karşılık 35.836
pencere); bu asimetri Bölüm 3.5'teki madenciliği ek maliyet olmaksızın mümkün
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

## 3.3 Varış Sabitleme

Varışlar toplanmamakta, **kestirilmektedir**. Her (olay, istasyon) çifti için
episantr uzaklığı katalog hiposantrı ve istasyon koordinatlarından hesaplanmakta;
ilk varan P fazı (`p`, `P`, `Pg`, `Pn`) iasp91 ile TauP kullanılarak
belirlenmektedir. Pencere, kestirilen varıştan pencere uzunluğunun üçte biri
kadar önce başlatılmaktadır (6 s için 2,0 s, 3 s için 1,0 s). **Hiçbir
tetikleyici ve eşik uygulanmamaktadır**; sessiz olduğu için elenen kayıt yoktur.

**Çizelge 2.** Kestirilen varışların, varışı görebilecek kadar kısa bir LTA ile
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

## 3.4 Bölümleme

Bölümler **istasyon-ayrıktır**: her istasyon, her iki sınıf için de tek bir bölüme
atanmaktadır; eğitimde görülen bir istasyonun test kümesinde herhangi bir etiket
altında yeniden görünmesi yapısal olarak olanaksızdır. Atama tohumlanmış olup
üretim tekrarlanabilirdir.

**Çizelge 3.** İstasyon-ayrık bölümleme (6 s).

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

## 3.5 Zor Negatif Madenciliği

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

## 3.6 Model ve Eğitim

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

**Çizelge 4.** Parametre sayıları ve eğitim yapılandırması.

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

## 3.7 Değerlendirme Ölçütleri

Birincil ölçüt ROC-AUC'dir; eşikten bağımsız olduğu için sabit 0,5 kesme noktası
kullanan bir modelle eşik seçimi yapan bir tabanın adil karşılaştırılmasına izin
vermektedir. Ayrıca accuracy, MCC, PR-AUC ve Brier skoru raporlanmaktadır.

## 3.8 Karşılaştırma Tabanları

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

## 3.9 Karma Duyarlıkta Genlik Kanalının Sayısal Güvenliği

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

Bu bölümün başlıca sonuçları Çizelge 5'te toplanmıştır. Her satırda
bildirilen katkı, o satırın kendi koşullu tabanına göredir; **kazanılan pay**,
taban ile 1,0 arasındaki açıklığın kapatılan oranıdır ve farklı tabanlara sahip
kurulumları karşılaştırmanın tek doğru yoludur.

**Çizelge 5.** Başlıca sonuçlar.

| Sonuç | Değer | Taban | Katkı | Kazanılan pay | Bölüm |
|---|---|---|---|---|---|
| **En iyi yapılandırma** — birleştirme, zor negatif | **0,9908** | 0,9049 | **+0,0859** | **%90,3** | 4.8 |
| Yalnız 1B (evrişim + BiLSTM), zor negatif | 0,9896 | 0,9049 | +0,0847 | %89,1 | 4.6 |
| Yalnız 2B, zor negatif | 0,9882 | 0,9049 | +0,0833 | %87,6 | 4.2 |
| Korpuslar arası (eşleşmiş STEAD) | 0,9971 | 0,9752 | +0,0219 | %88,3 | 4.4 |
| 3 s pencere, katalog sabitli | 0,9805 | 0,8481 | +0,1324 | %87,2 | 4.5 |
| 3 s pencere, kayan pencere | 0,9107 | 0,7412 | +0,1695 | %65,5 | 4.5 |

Çizelge 5 yalnızca bu çalışmanın kendi ölçümlerini içermektedir. Karşılaştırma
olarak, önceden eğitilmiş EQTransformer (SeisBench, `instance` ağırlıkları)
aynı 27.378 iz üzerinde iki uç yapılandırmada 0,9989 (tam 60 s iz) ve 0,9565
(girdi 6 s pencereye maskelenmiş) vermektedir. Bu aralık yukarıdaki 0,9971
değerini içine aldığından ölçüm bir sıralama vermemektedir; ayrıntısı Bölüm
4.4 ve Çizelge 11'dedir.

Bu sonuçların yorumu ve yöntem bakımından taşıdıkları anlam Bölüm 5'te ele
alınmaktadır.

## 4.1 Koşullu Tabanların Ölçülmesi

**Çizelge 6.** Test bölümünde öğrenmesiz tekil istatistikler (n = 9.548).

| İstatistik | Gürültü (medyan) | Olay (medyan) | ROC-AUC |
|---|---|---|---|
| `img` ortalama dB | −0,48 | 18,77 | 0,9205 |
| log SNR | −1,138 | 2,009 | 0,9404 |
| `seq` mutlak maksimum | 1,535 | 44,379 | **0,9461** |
| Çoğunluk sınıfı | — | — | 0,5000 |

Aynı pencereler üzerinde STA/LTA, sabitleme geometrisine uygun parametrelerle
(STA 0,03 s / LTA 0,3 s; doğrulamada seçilmiş) **0,8193** vermektedir.

**Çizelge 7.** Veri kümesi kurulumuna göre koşullu tabanlar.

| Veri kümesi | `seq` mutlak maks. | `img` ortalama dB | Taban |
|---|---|---|---|
| Özgün (STA/LTA kapılı) | 0,9461 | 0,9208 | 0,9461 |
| Katalog sabitli, rastgele gürültü | 0,9535 | 0,8613 | 0,9535 |
| Katalog sabitli + zor negatif | **0,9049** | **0,7571** | **0,9049** |

Seçim kapısının kaldırılması tek başına tabanı düşürmemiş; yalnızca negatiflerin
madenlenmesi tabanı hareket ettirmiştir.

## 4.2 Alan İçi Başarım

**Çizelge 8.** Yapılandırmalara göre test ROC-AUC değerleri.

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
ölçülmüştür (bkz. Bölüm 3.9). † satırları yeniden ölçülmemiştir; bu iki satır `seq`
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

## 4.3 Gürültü Rejimleri Arası Aktarım

Her iki küme olaylar, bölümler ve istasyon ataması bakımından özdeş olduğundan
yalnızca zor negatiflerin etkisi yalıtılmaktadır.

**Çizelge 9.** Gürültü rejimleri arası çapraz değerlendirme (ROC-AUC).

| Eğitim ↓ / Değerlendirme → | Rastgele gürültü (taban 0,9535) | Zor negatif (taban 0,9049) |
|---|---|---|
| Rastgele gürültü | 0,9885 (+0,0350) | **0,9841 (+0,0792)** |
| Zor negatif | 0,9873 (+0,0338) | **0,9896 (+0,0847)** |

Yalnızca rastgele gürültüyle eğitilen model, hiç görmediği yüksek genlikli
gürültü geçicilerinde 0,9841 vermektedir; aynı kümede genlik skaleri 0,9049'dur.
Zor negatif kümesi üzerinde iki eğitim rejiminin katkıları +0,0792 ve +0,0847,
aradaki fark 0,0055'tir.

## 4.4 Korpuslar Arası Genelleme (STEAD)

Yeniden eğitim ve ince ayar yapılmaksızın model doğrudan uygulanmıştır.

**Çizelge 10.** STEAD üzerinde başarım.

| Eğitim verisi | Değerlendirme | n | AUC | Taban | Katkı |
|---|---|---|---|---|---|
| Kapılı (P içermeyen pencereler) | Eşleşmiş | 27.378 | 0,9818 | 0,9752 | +0,0066 |
| **Katalog sabitli** | **Eşleşmiş** | 27.378 | **0,9971** | 0,9752 | **+0,0219** |
| Kapılı (P içermeyen pencereler) | Tam aralık | 50.000 | 0,9235 | 0,9531 | −0,0296 |
| **Katalog sabitli** | **Tam aralık** | 50.000 | **0,9693** | 0,9531 | **+0,0162** |

Büyüklüğe göre ayrıştırıldığında (katalog sabitli model, tam STEAD) AUC değerleri
M < 1,0 için 0,9482 (n = 11.029), 1,0–1,5 için 0,9747 (n = 6.038), 1,5–2,0 için
0,9922 (n = 3.752), 2,0–2,5 için 0,9964 (n = 2.235), 2,5–3,0 için 0,9968
(n = 871) ve M ≥ 3,0 için 0,9972'dir (n = 1.074). Eğitim korpusunun en küçük
büyüklüğü M 2,00'dir; AUC değerleri bunun altındaki büyüklüklere doğru tekdüze
azalmaktadır.

**Öğrenen bir dedektörle karşılaştırma.** Aynı 27.378 iz üzerinde, önceden
eğitilmiş EQTransformer (SeisBench, `instance` ağırlıkları) çalıştırılmıştır.
`original` ve `stead` ağırlıkları STEAD üzerinde eğitildiğinden, `scedc` ise
STEAD'in Güney Kaliforniya içeriğiyle örtüştüğünden kullanılmamıştır.

İki model farklı girdi uzunluklarıyla çalışmaktadır: bu projenin dedektörü 600
örnek (6 s), EQTransformer 6000 örnek (60 s) görmektedir. Bu nedenle iki uç
yapılandırma ölçülmüştür.

**Çizelge 11.** EQTransformer'ın eşleşmiş STEAD kümesindeki değerleri
(n = 27.378, taban 0,9752).

| Yapılandırma | AUC | Katkı |
|---|---|---|
| EQTransformer, tam 60 s iz | 0,9989 | +0,0237 |
| Bu proje, 6 s pencere | 0,9971 | +0,0219 |
| EQTransformer, yalnızca 6 s pencere (girdi maskeli) | 0,9565 | −0,0187 |

Üst satırda EQTransformer tasarlandığı biçimde, tam iz üzerinde
çalıştırılmıştır; bu kurulumda P ve S varışları ile koda birlikte
görülmektedir. Alt satırda pencere dışındaki örnekler sıfırlanarak iki modele
aynı bilgi verilmiştir; bu maske EQTransformer için dağılım dışıdır ve değeri
düşürmektedir.

İki değer arasındaki aralık (0,9565–0,9989) bu projenin sonucunu içine
almaktadır.

> **Uyarı.** STEAD gürültüsü genlik ölçeğinde bu korpusun gürültüsünün ~2 katıdır
> (medyan `seq` std 0,98'e karşılık 0,47). STEAD *içindeki* sıralama etkilenmediği
> için ROC-AUC ve PR-AUC aktarılmakta; accuracy, MCC ve Brier skoru
> aktarılmamakta ve yeniden kalibrasyon gerektirmektedir.

## 4.5 Pencere Uzunluğu: 3 s ve 6 s

Pencere uzunluğunu yalıtmak için her iki yapılandırma **aynı yordamla** yeniden
üretilmiştir: aynı katalog sabitleme, aynı normalizasyon, aynı zor negatif
madenciliği ve aynı STFT parametreleri (n_fft = 64, hop = 16). Karşılaştırmaya,
3 s pencerelerin 6 s kayıtlardan %50 örtüşmeli **kayan pencere** ile türetildiği
üçüncü bir küme de dâhil edilmiştir; bu kümede varış t = 2,0 s'de bulunduğundan
üç alt pencerenin yalnızca ikisi başlangıcı içermektedir.

**Çizelge 12.** Pencere uzunluğu ve çıkarım yöntemine göre başarım.

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

## 4.6 1B Kolunun Mimarisi

Bölüm 4.2'deki 1B sonuçları, 600 ham örneği doğrudan bir BiLSTM'e veren, yani
**hiçbir evrişimli ön uç içermeyen** bir kolla elde edilmiştir. Karşılaştırma
konusu güncel dedektörler bunun tersini yapmaktadır: EQTransformer
CNN → BiLSTM → dikkat sırasını, PhaseNet ise 1B evrişimlerden oluşan bir U-Net'i
kullanmaktadır. Kol bu nedenle zor negatif kümesi üzerinde üç yapılandırmada
yeniden sınanmıştır. Karşılaştırmanın tek değişkenli kalması için spektrogram
kolu tümüyle devre dışı bırakılmış (`--channels 1d`), üç yapılandırma da aynı
tohumlar, aynı bölümleme ve aynı asinh dönüşümü ile çalıştırılmıştır.

**Çizelge 13.** 1B kol mimarisine göre başarım (zor negatif kümesi, taban
0,9049, n = 15.812).

| Yapılandırma | Tohum başına | Ort. | Std | Topluluk | Katkı | Parametre |
|---|---|---|---|---|---|---|
| **Evrişim + BiLSTM** | 0,9891 / 0,9882 / 0,9892 | **0,9888** | 0,0005 | **0,9896** | **+0,0847** | 142.059 |
| Yalnız BiLSTM | 0,9869 / 0,9876 / 0,9874 | 0,9873 | 0,0003 | 0,9883 | +0,0834 | 76.707 |
| Yalnız evrişim | 0,9818 / 0,9826 / 0,9836 | 0,9827 | 0,0007 | 0,9843 | +0,0794 | 48.555 |

**Çizelge 14.** Aynı yapılandırmaların hata bileşimi (eşik 0,5; 7.906 olay ve
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

## 4.7 İşletim Zarfı: Recall Neye Bağlıdır?

Dedektörün bulduğu ve kaçırdığı olaylar, kaynak parametrelerine göre
ayrıştırılmıştır. Tespit kümesindeki her olay penceresi büyüklük kümesinin
künyesine dosya adıyla eşlenmektedir (7.906 / 7.906); künye büyüklük, log SNR
ve merkez üssü uzaklığı taşımaktadır.

Genel recall **0,9484**'tür (7.498 / 7.906; 408 kaçırılan olay). Bu bölümün
değerleri, aynı üç ağırlık dosyası ayrı bir betikle yeniden puanlanarak
üretilmiştir; ROC-AUC Çizelge 13'teki değerle özdeştir (0,9896), 0,5 eşiğinin
iki yanında kalan pencere sayısı ise iki pencere farkla Çizelge 14'ün 410
değeri yerine 408 çıkmaktadır.

**Çizelge 15.** Kaynak parametrelerine göre recall (evrişim + BiLSTM topluluğu,
eşik 0,5).

| Büyüklük | n | Recall | | log SNR | n | Recall | | Uzaklık (km) | n | Recall |
|---|---|---|---|---|---|---|---|---|---|---|
| 2,0 | 1.611 | 0,9100 | | < −2,0 | 35 | 0,8857 | | 0–25 | 1.588 | 0,9773 |
| 2,0–2,5 | 4.152 | 0,9458 | | −2,0 – 0,72 | 2.630 | 0,8696 | | 25–50 | 5.044 | 0,9417 |
| 2,5–3,0 | 1.445 | 0,9792 | | 0,72 – 3,42 | 4.261 | 0,9859 | | 50–100 | 1.274 | 0,9388 |
| 3,0–3,5 | 498 | 0,9900 | | > 3,42 | 976 | **1,0000** | | | | |
| > 3,5 | 200 | 0,9850 | | | | | | | | |

**Çizelge 16.** Kaçırılan ve bulunan olayların medyan kaynak parametreleri.

| | Kaçırılan (n = 408) | Bulunan (n = 7.498) | Fark |
|---|---|---|---|
| **log SNR** | **−0,006** | **1,431** | **1,437** |
| Büyüklük | 2,100 | 2,300 | 0,200 |
| Uzaklık (km) | 43,06 | 38,47 | 4,59 |

![Sekil 1](/home/hogib/Desktop/rapor_sekiller/sekil1_isletim_zarfi.png)

**Şekil 1.** Recall'ın log SNR ve büyüklüğe göre değişimi (evrişim + BiLSTM topluluğu, eşik 0,5). Kesikli çizgi genel recall değeridir (0,9484). Sol panelde recall tekdüze ve dik biçimde artmakta; sağ panelde görülen eğilim ise büyük ölçüde SNR'nin dolaylı yansımasıdır.

Kaçırılan ve bulunan olaylar arasındaki medyan farklar Çizelge 16'da
verilmiştir: log SNR'de 1,44, büyüklükte 0,2 birim, uzaklıkta 4,6 km. Recall,
log SNR 0,72'nin altında 0,87, üstünde 0,986 ve 3,42'nin üzerinde 976 olayda
1,0000'dır. Uzaklık ölçümü bu kümede 100 km ile sınırlıdır.

## 4.8 İki Kolun Birleştirilmesi

Birleştirme, zor negatif kümesinde iki 1B kolu yapılandırmasıyla ölçülmüştür:
evrişimli ön uçlu ve ön uçsuz. İkinci yapılandırma, birleştirmenin katkısının
güçlü koldan mı yoksa birleştirmenin kendisinden mi geldiğini ayırmak için
çalıştırılmıştır.

**Çizelge 17.** Birleştirilmiş modelin başarımı (doğrusal birleştirme, zor
negatif kümesi, taban 0,9049).

| Yapılandırma | Tohum başına | Ort. | Std | Topluluk | Katkı |
|---|---|---|---|---|---|
| Birleştirme (1B: evrişim + BiLSTM) | 0,9902 / 0,9905 / 0,9897 | 0,9901 | 0,0003 | **0,9908** | **+0,0859** |
| Birleştirme (1B: yalın BiLSTM) | 0,9901 / 0,9904 / 0,9898 | 0,9901 | 0,0002 | 0,9907 | +0,0858 |
| Yalnız 1B (evrişim + BiLSTM) | 0,9891 / 0,9882 / 0,9892 | 0,9888 | 0,0005 | 0,9896 | +0,0847 |
| Yalnız 2B | 0,9876 / 0,9874 / 0,9864 | 0,9871 | 0,0005 | 0,9882 | +0,0833 |

**Çizelge 18.** Hata bileşimi (eşik 0,5; 7.906 olay ve 7.906 gürültü penceresi).

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

## 4.9 Genlik Silindiğinde Geriye Ne Kalıyor?

Bölüm 4.2'nin en güçlü olumsuz sonucu, 1B kolunun genliğin ötesinde katkı
sağlamadığıdır. Bu sonuç, evrişimsiz bir kolla ölçülmüştür. Aynı ölçüm burada
Bölüm 4.6'nın mimarileriyle yinelenmektedir.

Ölçüm için pencere bazlı normalize küme kullanılmaktadır: standartlaştırma her
pencereyi kendi istatistiklerine göre ölçeklediğinden **mutlak genlik
silinmektedir**. Bunun sonucu tabanlarda doğrudan görülmektedir.

**Çizelge 19.** Pencere bazlı normalize kümede öğrenmesiz tabanlar
(n = 9.548).

| İstatistik | ROC-AUC |
|---|---|
| `seq` std | **0,5000** — tam rastlantı |
| `seq` mutlak maksimum | 0,7088 |
| `img` ortalama dB | **0,9205** — en güçlü önemsiz taban |

`seq` std tabanı tam olarak 0,5000'dir; `seq` mutlak maksimum tabanı
0,7088'e inmektedir. Bu kümede 1B kolunun girdisinde mutlak genlik
bulunmamaktadır.

**Çizelge 20.** 1B mimarilerine göre başarım (pencere bazlı normalize küme).

| Yapılandırma | Tohum başına | Ort. | Std | Topluluk | `seq` tabanına göre | `img` tabanına göre |
|---|---|---|---|---|---|---|
| **Evrişim + BiLSTM** | 0,9301 / 0,9237 / 0,9278 | 0,9272 | 0,0026 | **0,9309** | **+0,2221** | **+0,0104** |
| Yalnız evrişim | 0,9090 / 0,9126 / 0,9084 | 0,9100 | 0,0019 | 0,9146 | +0,2058 | −0,0059 |
| Yalın BiLSTM | 0,9139 / 0,9116 / 0,9110 | 0,9122 | 0,0013 | 0,9165 | +0,2077 | −0,0040 |

Üç yapılandırma da `seq` tabanını (0,7088) 0,20'nin üzerinde geçmektedir.
Yalın BiLSTM (0,9165) ve yalnız evrişim (0,9146) `img` tabanının (0,9205)
altında, evrişim + BiLSTM (0,9309) üzerindedir.

**Genlik korunduğunda aynı mimariler.** Aynı üç mimari, genliğin korunduğu
özgün küme üzerinde de ölçülmüştür (Çizelge 21). Sonuç, yukarıdakinin karşıt
denetimidir.

**Çizelge 21.** Genlik korunmuş özgün kümede 1B mimarileri (taban 0,9461).

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

**Çizelge 22.** Pencere bazlı normalize kümede kanal yapılandırmaları.

| Yapılandırma | Tohum başına | Ort. | Std | Topluluk | `img` tabanına göre |
|---|---|---|---|---|---|
| Yalnız 2B | 0,9770 / 0,9776 / 0,9799 | 0,9782 | 0,0013 | **0,9794** | +0,0589 |
| Birleştirme, doğrusal (1B: evrişim + BiLSTM) | 0,9681 / 0,9646 / 0,9695 | 0,9674 | 0,0021 | 0,9692 | +0,0487 |
| Yalnız 1B (evrişim + BiLSTM) | 0,9301 / 0,9237 / 0,9278 | 0,9272 | 0,0026 | 0,9309 | +0,0104 |

Birleştirme, 1B kolunu tek başına 0,038 geçmekte; ancak 2B kolunun **0,0102
altında** kalmaktadır. Özgün küme üzerinde ölçülen iki birleştirme türüyle
(Çizelge 8) birlikte değerler aşağıdaki gibidir.

**Çizelge 23.** Birleştirilmiş modelin tekil 2B koluna göre farkı, dört veri
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

Çizelge 8'de bu satırın tabanı olarak kümenin en güçlü öğrenmesiz
istatistiği (`img` ortalama dB, 0,9205) kullanılmıştır. Çizelge 20 ayrıca kolun
kendi kanalının tabanına (`seq` mutlak maksimum, 0,7088) göre değerleri de
vermektedir. İki tabanın seçimi Bölüm 5.1'de tartışılmaktadır.

## 4.10 Olasılık Kalibrasyonu ve Eşik

Birleştirilmiş modelin (evrişim + BiLSTM, zor negatif) ürettiği olasılıklar,
topluluk logit ortalaması üzerinden ölçülmüştür. Sıcaklık ölçekleme parametresi
**doğrulama** bölümünde uyarlanmış, test bölümüne uygulanmıştır. Bu bölümdeki
topluluk logit uzayında, Çizelge 17 ve 18'deki ise olasılık uzayında
alınmaktadır; 0,5 eşiğinde iki yöntem sırasıyla 69 ve 75 yanlış alarm
vermektedir.

**Çizelge 24.** Kalibrasyon ölçütleri (test, n = 15.812).

| | ECE | MCE | Brier |
|---|---|---|---|
| Kalibrasyon öncesi | 0,0863 | 0,1904 | 0,0325 |
| Kalibrasyon sonrası (T = 0,476) | **0,0216** | 0,3353 | **0,0269** |

Uyarlanan sıcaklık 1'in altındadır. Kalibrasyon öncesinde 0,9–1,0 kutusundaki
ortalama olasılık 0,901 iken bu kutudaki gerçek olay oranı 0,999; 0,0–0,1
kutusunda ise ortalama olasılık 0,097 iken gerçek oran 0,012'dir.

MCE değeri kalibrasyon sonrasında yükselmektedir. Ölçekleme, kütlenin neredeyse
tamamını uç kutulara taşımakta; kalan orta kutularda 15.812 pencerenin yalnızca
26–100'ü bulunmaktadır. ECE kutuları doluluklarına göre ağırlıklandırmakta, MCE
ise ağırlıklandırmamaktadır.

**Çizelge 25.** Kalibre edilmiş olasılıklarla eşik seçenekleri.

| Eşik | MCC | Recall | Precision | Yanlış alarm |
|---|---|---|---|---|
| 0,28 (MCC enbüyük) | 0,9402 | 0,9526 | 0,9866 | 102 |
| 0,50 | 0,9391 | 0,9469 | 0,9909 | 69 |
| 0,70 | 0,9356 | 0,9411 | 0,9929 | 53 |
| 0,90 | 0,9274 | 0,9284 | 0,9966 | 25 |

# 5. TARTIŞMA

## 5.1 Karşılaştırma Tabanının Seçimi Sonucu Belirlemektedir

Bu çalışmanın en aktarılabilir bulgusu, bir modelin başarımının değil, o başarımın
**neye göre** ölçüldüğünün belirleyici olduğudur. En iyi yapılandırmanın
0,9908'lik ROC-AUC değeri, çoğunluk sınıfı tabanına göre 0,491'lik bir katkı
gibi görünmektedir; en güçlü koşullu tabana göre ise katkı **0,0859**'dur.
Aradaki fark yaklaşık bir büyüklük mertebesindedir ve modelden değil, veri
kümesinin kuruluş biçiminden kaynaklanmaktadır.

Tabanın ikinci ve daha az belirgin bir işlevi Bölüm 4.9'da ortaya çıkmaktadır:
taban yalnızca katkının **büyüklüğünü** değil, bir farkın **ölçülebilir olup
olmadığını** da belirlemektedir. Genlik korunmuş kümede `seq` tabanı 0,9461'dir
ve 1,0'e yalnızca 0,0539 açıklık bırakmaktadır; bu kurulumda üç ayrı 1B
mimarisi arasındaki fark ölçülememektedir (ortalamalar 0,9416–0,9428). Genlik
silindiğinde aynı taban 0,7088'e düşmekte, açıklık 0,2912'ye çıkmakta ve aynı
mimariler arasında 0,0165'lik bir fark görünür hale gelmektedir. **Doyuma yakın
bir taban, gerçek bir farkı yokmuş gibi gösterebilmektedir.** Bölüm 4.3'ün
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

## 5.2 Mimari Katkısı Küçüktür — Ancak Sıfır Değildir ve Nerede Aranacağı Belirlenmiştir

Bu çalışmanın ilk turunda sınanan yapısal eklentiler — çift kanallı mimari,
kapılı birleştirme, geç birleştirmeli yığınlama — yalın spektrogram CNN'ini
geçememişti. Bölüm 4.6'daki ölçüm bu sonucu bütünüyle geçersiz kılmamakta,
ancak önemli ölçüde niteliklendirmektedir: eklentinin **nereye** yapıldığı
belirleyicidir.

1B kolu içinde mimari fark ölçülebilirdir. Yinelemesiz bir evrişim yığını
0,9843'te kalırken, yineleme içeren yapılandırmalar 0,9883 ve 0,9896'ya
ulaşmaktadır; aradaki ~0,005'lik fark tohum yayılımının yaklaşık on katıdır.
Bu farkın tümü **recall'dan** kaynaklanmaktadır (kaçırılan olay 664 → 410,
precision değişmeksizin). Dolayısıyla "mimari önemsizdir" biçimindeki bir
genelleme desteklenmemektedir; desteklenen ifade, mimari kazanımlarının
0,005 mertebesinde kaldığı, veri kümesi kurulumundan kaynaklanan kazanımların
ise (Bölüm 5.3) bir mertebe büyük olduğudur.

1B kolunun katkısının genlikle sınırlı olduğu yönündeki daha güçlü sonuç ise
Bölüm 4.9'da **doğrudan sınanmış ve geçerliliğini korumamıştır**. Genlik
tümüyle silindiğinde (`seq` std tabanı tam 0,5000) evrişimli ön uçlu yinelemeli
kol 0,9309'a ulaşmakta ve hiç görmediği spektrogram tabanını dahi
geçmektedir; yeniden öğrenilecek bir skaler yokken bu başarım ancak dalga
biçimi karakterinden gelebilir. Söz konusu olan, kolun ulaşamadığı bir bilgi
değil, evrişimsiz bir yinelemenin erişemediği bir gösterimdir.

Birleştirme konusundaki daha önceki sonuç da (Çizelge 8) benzer
biçimde niteliklendirilmelidir. Zor negatif kümesinde doğrusal birleştirme her
iki kolu da geçmekte (0,9908; kollar 0,9896 ve 0,9882) ve yanlış alarm sayısını
her iki kolun da altına indirmektedir (64–75; kollar 141 ve 90). Bölüm 4.2'de
ölçülen kötüleşme özgün küme üzerinde ve **yalnızca kapılı** birleştirmeyle
elde edilmiştir; dolayısıyla "birleştirme zarar vermektedir" ile "kapı zarar
vermektedir" ayrımı o ölçümle yapılamamaktadır. Bu ayrım özgün küme üzerinde ölçülmüştür (Çizelge 8): kapılı birleştirme
0,9745, doğrusal birleştirme 0,9730 vermekte, ikisi de tekil 2B kolunun
0,9779 değerinin altında kalmaktadır. Kötüleşme kapıya özgü değildir.

Buna karşılık mimari katkının **nerede** ölçüldüğü belirleyicidir: 1B kolu tek
başına çalışırken evrişimli ön uç 0,0015 kazandırırken, spektrogram kolu
devredeyken bu kazanç ölçülemez düzeye inmektedir (Çizelge 17: 0,9908'e karşılık
0,9907). Tek
kollu ablasyonlarda anlamlı görünen bir mimari farkın tam modelde
kaybolabileceği, bu çalışmanın yöntemsel çıkarımlarından biridir.

Eğitim örneği başına ~1,5 parametre ile model kapasite sınırlı değildir; bu
gözlem geçerliliğini korumaktadır.

## 5.3 Veri Kümesi Kurulumu Mimariden Daha Belirleyicidir

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

## 5.4 İşletim Sınırını Büyüklük Değil SNR Belirlemektedir

Bölüm 4.7'deki ölçüm, dedektörün neyi bulup neyi kaçırdığını tek bir AUC
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

## 5.5 Eşik Bir Tasarım Kararıdır

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

## 5.6 Sınırlılıklar

1. **Pozitif sınıfta artık etiket gürültüsü.** Katalog, bir depremin
   gerçekleştiğini bildirmektedir; *bu istasyonun* onu kaydettiğini değil.
   Ölçümler pozitiflerin ~%10–15'inin sınırda olduğunu göstermektedir;
   ulaşılabilir tavan 1,0'in altındadır.
2. **Varış doğruluğu yalnızca tespit içindir** (0,63 s medyan mutlak sapma).
3. **Zor negatif kümeleri bilinçli olarak temsili değildir**; kalibre edilmiş
   çalışma noktası değerleri rastgele örneklenmiş test kümesinden alınmalıdır.
4. **Uzaklık aralığı dardır.** İndirme yarıçapı 0,5° olduğundan episantr
   uzaklığı ~56 km ile sınırlıdır (Çizelge 1: p95 53,5 km, maks 55,6 km) ve
   dedektörleri en çok ayrıştıran düşük SNR'li uzak rejim bu aralığın dışında
   kalmaktadır. Bölüm 4.7'de uzaklığın recall üzerindeki zayıf etkisi
   (0,9773 → 0,9388) bu aralık içinde geçerlidir; genel bir sönümlenme ifadesi
   değildir.
5. **Güncel dedektörlerle karşılaştırma kısmen yapılmıştır.** EQTransformer
   (SeisBench, `instance` ağırlıkları) eşleşmiş STEAD kümesinde iki uç
   yapılandırmada çalıştırılmıştır; elde edilen aralık (0,9565–0,9989) bu
   projenin değerini içine almakta, dolayısıyla ölçüm bir sıralama
   vermemektedir (Çizelge 11). İki modelin girdi uzunlukları ve tasarım
   amaçları farklı olduğundan aradaki farkın ne kadarının mimariden
   geldiği bu kurulumla ayrılamamaktadır. PhaseNet ve GPD henüz
   çalıştırılmamıştır; ayrıca karşılaştırma yalnızca STEAD üzerinde yapılmış
   olup bu korpusun kendi ölçüt kümesinde yinelenmesi gerekmektedir.
6. **Tek bölge, tek katalog** ile eğitim yapılmıştır. Sonuçların başka bir
   sismotektonik ortama ve başka bir katalog uygulamasına aktarılabilirliği
   ölçülmemiştir; Bölüm 4.4'teki STEAD değerlendirmesi bu soruyu yalnızca
   kısmen yanıtlamaktadır. Olasılıklar Bölüm 4.10'da sıcaklık ölçeklemeyle
   kalibre edilmiştir; ölçekleme parametresi bu korpusa özgüdür ve başka bir
   korpusa aktarılmamalıdır.
7. **Mixed precision'da sayısal taşma riski (Bölüm 3.9) taranmış ve
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

## 5.7 Sonraki Adımlar

Öncelik sırasıyla:

1. **PhaseNet ve GPD ile karşılaştırma**, ve her üç dedektörün bu korpusun kendi
   ölçüt kümesi üzerinde çalıştırılması (Sınırlılık 5). Şu ana kadarki
   karşılaştırma yalnızca EQTransformer ile ve yalnızca STEAD üzerindedir.
2. **İndirme yarıçapının genişletilmesi**, düşük SNR'li uzak rejimin kapsanması
   (Sınırlılık 4). Bölüm 4.7 işletim sınırını SNR'nin belirlediğini gösterdiği
   için, ayırt ediciliğin en çok sınanacağı aralık bugün veri kümesinde
   bulunmamaktadır.
3. **3 s yapılandırmasının STEAD üzerinde korpuslar arası sınanması.** Alan içi
   sonuç tamamlanmış (Bölüm 4.5), 6 s yapılandırmasının korpuslar arası
   davranışı ölçülmüştür (Bölüm 4.4); kısa pencerelerin genelleme davranışı
   henüz ölçülmemiştir.
4. **Sürekli veride yanlış alarm oranının doğrudan ölçülmesi** (Sınırlılık 8).
   Bugün bildirilen precision dengeli bir ölçüt kümesine aittir; konuşlandırma
   için gereken sayı kayan pencereli sürekli kayıttan ölçülmelidir.

---

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
