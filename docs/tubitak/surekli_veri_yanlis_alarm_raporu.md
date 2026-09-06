# Sürekli Veri Üzerinde Kısa Pencereli Deprem Tespitinde Yanlış Alarm Oranları

MANT ve GCAM istasyonlarına ait sürekli kayıt üzerinde üç sinir ağı ve bir klasik eşik yönteminin karşılaştırılması; tek istasyon yanlış alarm oranları ve iki istasyon uyuşmasının bunlara etkisi

---

## Öz

Kısa pencereli deprem/gürültü sınıflandırıcıları, dengeli sınıflardan ve varış zamanına sabitlenmiş pencerelerden oluşan **düzenlenmiş sınama kümelerinde** değerlendirilmektedir. Bu değerlendirme koşullu bir niceliği ölçmektedir: pencerede bir varış olduğu bilindiğinde model onu bulabiliyor mu? Sürekli işletim ise koşulsuz niceliği gerektirmektedir ve iki durumdaki taban oranlar arasında dört ila altı büyüklük mertebesi fark bulunmaktadır.

Bu çalışmada, projede geliştirilen üç tespit edici ile klasik bir STA/LTA yöntemi, MANT istasyonuna ait 30 Nisan 2024 – 9 Ağustos 2026 arasındaki 728 günlük kesintisiz kayıt üzerinde, aynı pencerelerde ve aynı ön işlemeyle karşılaştırılmıştır. Toplam 10,5 milyon (6 saniyelik model) ve 18,5 milyon (3,4 saniyelik modeller) pencere puanlanmıştır.

Dört ana bulgu elde edilmiştir. **(1)** Sınama kümesinde belirlenen 0,5 eşiği sürekli veriye aktarılamamaktadır: öngörülen günde 257 yanlış alarma karşılık ölçülen değer 12.599'dur, çünkü 6 saniyelik model sürekli gürültü üzerinde 0,80 ortanca puan vermektedir. **(2)** Nedeni, genlik madenciliğinin negatif sınıfa koyduğu taban ile fiziğin pozitif sınıfa koyduğu taban arasında kalan ve modelin hiç eğitim verisi görmediği bir **sessizlik bölgesidir**; sürekli arka plan tam olarak orada bulunmaktadır. **(3)** Olay düzeyinde AUC ile işletme çalışma noktası, dört tespit ediciyi **ters sırada** dizmektedir. **(4)** Açıklanamayan alarmlar günlük döngü göstermektedir (gündüz/gece 1,7–2,1 kat), bu da önemli bir bölümünün kataloglanmamış deprem değil kültürel gürültü olduğunu göstermektedir. **(5)** Yanlış alarm oranı **istasyona özgüdür**: aynı model, aynı ön işlemeyle, MANT'ta 0,80 olan arka plan ortancasını GCAM'de 0,12 vermektedir. **(6)** İki istasyonun uyuşması şart koşulduğunda kazanç **istasyon uzaklığına bağlıdır**: 63 km aralıklı çiftte yanlış alarmlar bağımsızlık varsayımının öngördüğünün 29–58 katı örtüşmekte, 144 km aralıklı çiftte ise varsayım geçerli olmaktadır (1,1–2,2 kat). Uzak çift, **aynı duyarlılıkta on kat daha az** yanlış bildirim üretmektedir — 23 günde bir, 2,4 güne karşılık.

Ayrıca P varışının S dalgasından önce bildirilebilme oranı uzaklığa göre ölçülmüş ve bu yeteneğin pencere boyunun dayattığı bir **başabaş uzaklığa** bağlı olduğu gösterilmiştir.

**Anahtar sözcükler:** deprem tespiti, sürekli veri, yanlış alarm oranı, çalışma noktası, ağ uyuşması, istasyonlar arası aktarım, STA/LTA, derin öğrenme

---

## 1. Giriş

Derin öğrenme tabanlı faz seçiciler, sismolojide rutin katalog üretiminin standart aracı hâline gelmiştir. Buna karşılık, bu modellerin **sürekli kayıt üzerindeki yanlış alarm maliyeti** yayımlanmış çalışmalarda nadiren raporlanmaktadır.

Bu boşluk alan yazında açıkça kabul edilmektedir. PhaseNet'in özgün makalesinde Zhu ve Beroza (2019), modellerinin "tespit edilmiş depremlerden oluşan bir veri kümesi üzerinde eğitildiğini" belirtmekte ve sürekli veride tespit için "**sismik faza benzeyen gürültü sıçramalarını ayırt edebilmesi amacıyla daha fazla sismik olmayan sinyal içeren yeni bir veri kümesinin** eğitimde kullanılması gerektiğini" ifade etmektedir. Aynı makaledeki sürekli veri gösterimi, sekiz olayın dalga formlarının üst üste bindirilmesiyle **yapay olarak** oluşturulmuştur.

Benzer biçimde Ravishan vd. (2026), Yeni Zelanda'da uç cihaz üzerinde çalışan hafif bir evrişimli ağ için, gürültü pencerelerinin "**deprem dalga formu kayıtlarından rastgele çıkarıldığını** ve dolayısıyla yalnızca küçük bir bölümünün trafik, kalibrasyon darbeleri veya taş ocağı patlatmaları gibi yüksek genlikli geçici sinyaller içerdiğini" sınırlılık olarak belirtmekte, bunun "gerçek zamanlı işletimde ara sıra hatalı sınıflandırmalara yol açabileceğini" eklemektedir.

Doğal sınıf dengesizliğini sınama aşamasında koruyan az sayıdaki çalışmadan biri olan TransQuake (Hu vd., 2021), negatif örneklerini düşük eşikli bir FilterPicker'ın yanlış tetiklemelerinden seçmiş ve yaklaşık 11:1 dengesizlik altında **0,712 kesinlik** bildirmiştir. Bu değer, dengeli sınama kümelerinde bildirilen 0,99 düzeyindeki başarımlarla aynı büyüklükte değildir.

Bu çalışmanın katkısı, tek bir istasyonun iki yıla yakın kesintisiz kaydı üzerinde, **ilişkilendirme öncesi** ve **günlük yanlış alarm bütçesi cinsinden** doğrudan ölçüm yapmaktır. Ayrıca aynı kayıt üzerinde klasik bir eşik yöntemi karşılaştırma tabanı olarak çalıştırılmıştır; bu, yayımlanmış karşılaştırmaların çoğunda eksik olan bir ögedir.

## 2. Veri

### 2.1 Sürekli dalga formu kaydı

MANT istasyonu (TU ağı; 38,4908 K, 28,5579 D; Kula, Manisa) kayıtları AFAD TDVMS üzerinden 21 günlük parçalar hâlinde indirilmiştir. Kullanılabilir kayıt 30 Nisan 2024 – 9 Ağustos 2026 arasını kapsamakta ve 36 parçadan oluşmaktadır; dört pencere kaynakta veri içermediğinden dışarıda kalmıştır. Üç bileşen (HHZ, HHN, HHE) 100 Hz örnekleme hızındadır.

Kayıt kesintisiz değildir: istasyon 21 günlük bir parça içinde yüzlerce kez düşüp geri gelmektedir. Bu nedenle pencereler yalnızca **üç bileşenin de kesintisiz kayda sahip olduğu** aralıklarda üretilmiş, hiçbir pencere veri boşluğu üzerinden kurulmamıştır.

### 2.2 Deprem katalogu ve tespit edilebilirlik

AFAD katalogunda istasyondan 500 km içinde ve ilgili dönemde 62.494 olay bulunmaktadır. Bunların 14.062'si veri boşluğuna denk gelmekte ve değerlendirme dışında bırakılmaktadır — kaydı bulunmayan bir olay kaçırılmış sayılamaz.

Kalan olaylar için sinyal-gürültü oranı (SGO), iasp91 hız modeliyle kestirilen P varışı çevresinde ölçülmüştür (sinyal penceresi [−1, +12] s, gürültü penceresi [−60, −10] s). 47.522 olay için ölçüm yapılabilmiştir. **Ortanca SGO 1,39'dur ve olayların yalnızca %27'si SGO 3 eşiğini aşmaktadır.** Yani katalogdaki tipik deprem MANT'ın ham kaydında görünür bir iz bırakmamaktadır.

Bu, değerlendirme için bağlayıcı bir kısıttır. Bütün katalog olaylarına karşı ölçüldüğünde olay düzeyinde AUC 0,67–0,73 çıkmaktadır; bu değer modelin değil, **katalogun erişiminin** ölçüsüdür. Bu nedenle bu çalışmadaki bütün duyarlılık değerleri, SGO≥3 koşulunu sağlayan **13.056 olay** üzerinde raporlanmaktadır.

### 2.3 İkinci ve üçüncü istasyonlar (GCAM, DEMI)

Ölçümlerin ikinci bir istasyonda yinelenebilmesi ve ağ uyuşmasının ölçülebilmesi
için iki istasyon daha aynı yöntemle indirilmiş ve puanlanmıştır.

| İstasyon | Konum | İlçe | Puanlanan gün | Ortanca SGO | SGO≥3 oranı |
|---|---|---|---|---|---|
| MANT | 38,4908 K / 28,5579 D | Kula, Manisa | 728,3 | 1,39 | %27,0 |
| DEMI | 39,0428 K / 28,7162 D | Demirci, Manisa | 472,1 | **2,60** | **%46,7** |
| GCAM | 37,7139 K / 27,2418 D | Kuşadası, Aydın | 172,8 | 1,06 | %11,7 |

Çizelge 3b. Üç istasyonun kayıt süresi ve kaydedilebilirliği.

GCAM 9 Aralık 2024'ten sonra kayıt üretmemektedir; aynı sayıda istek
karşılığında MANT'ın 728 gününe karşılık 172,8 gün elde edilmiştir. Bu tek
başına, bir kampanyaya girişmeden önce istasyon çalışma süresinin yoklanması
gerektiğini göstermektedir. DEMI ise MANT'ın kaydının neredeyse tamamıyla
örtüşmekte ve üç istasyon arasında en duyarlı olanıdır.

İstasyon çiftleri arasındaki uzaklıklar ve aynı anda kayıtta oldukları süreler:

| Çift | Uzaklık | Ortak süre | İki istasyonda birden SGO≥3 olay |
|---|---|---|---|
| MANT–DEMI | **63 km** | 400,0 gün | **4.645** |
| MANT–GCAM | **144 km** | 159,0 gün | 132 |
| DEMI–GCAM | **196 km** | 57,7 gün | 48 |

Çizelge 3c. İstasyon çiftleri. Bölüm 4.9'daki bütün uyuşma ölçümleri yalnızca
ortak sürede yapılmıştır; bir istasyonun kayıtta olmadığı bir aralıkta
diğerinin alarmını "doğrulanmamış" saymak ölçüm değil eksik veri olurdu.

Son sütun bu çalışmanın bağlayıcı kısıtıdır. Uyuşma kuralının duyarlılığı
yalnızca iki istasyonda **birden** kaydedilmiş olaylar üzerinde
değerlendirilebilir, ve bu sayı MANT–DEMI için 4.645 iken diğer iki çift için
48 ile 132 arasındadır.

## 3. Yöntem

### 3.1 Pencereleme

Pencereler ayrıktır (adım = pencere boyu): 6 saniyelik model için günde 14.400, 3,4 saniyelik modeller için 25.412 pencere. Bu seçim, alarm sayısının aynı zamanda **bağımsız karar sayısı** olmasını sağlamaktadır. Toplam 10.487.211 ve 18.519.887 pencere puanlanmıştır.

### 3.2 Ön işleme ve doğrulama

Ön işleme, eğitim hattının kendi işlem sırasıdır: iki kez eğilim giderme, %5 Hann sönümlemesi, 4. derece 1–45 Hz bant geçiren süzgeç ve istasyonun uzun dönem gürültü tabanına göre standartlaştırma. Tek fark, işlemlerin tek pencere yerine pencere yığınlarına uygulanmasıdır.

Bu eşdeğerlik varsayılmamış, ölçülmüştür:

- Yığın süzgeci, özgün tek pencere işlevine karşı en fazla **5,4·10⁻¹³** bağıl fark vermektedir.
- Gerçek veri kümesi tensörleri, taramanın kendi puanlama yolundan geçirildiğinde yayımlanmış değerleri yeniden üretmektedir: 6 saniyelik model için **0,9899** (yayımlanmış 0,9896), P-dalgası modeli için **0,8742** (yayımlanmış 0,8712).

İstasyon gürültü tabanı, kaydın kendisinden saatlik parçalar hâlinde hesaplanmıştır (Z: σ=963,4; N: σ=1186,9; E: σ=1182,5 sayım).

### 3.3 Karşılaştırılan tespit ediciler

Üç sinir ağı kolu aynı mimariyi kullanmaktadır: şeritli evrişimlerle 8 kat indirgeme, ardından çift yönlü LSTM ve dört başlı öz-dikkat, sonunda ortalama havuzlama ve tek çıkışlı ikili baş (gizli boyut 48). Kollar yalnızca eğitildikleri veri kümesiyle ayrılmakta ve her biri üç tohumun (42, 43, 44) olasılık ortalaması olarak kullanılmaktadır.

| Kol | Pencere | Negatif seçimi | Sınama AUC'si |
|---|---|---|---|
| 6s | [P−2,0; P+4,0] s | genlik eşleştirmeli | 0,9896 |
| ponly | [P−2,0; P+1,4] s | genlik eşleştirmeli | 0,8712 |
| pnat | [P−2,0; P+1,4] s | madencilik yok | — |
| stalta | 6,0 s | — | — |

Çizelge 1. Karşılaştırılan tespit edici kolları.

6 saniyelik pencerede S dalgası olayların %28,8'inde, 25 km içindeki olayların %99,3'ünde pencere içine düşmektedir. P-dalgası kollarında 1,4 saniyelik kuyruk S dalgasını hız modeline göre dışarıda bırakmaktadır; bu kol bu nedenle bir **faz tespiti** görevi olarak okunmalıdır.

### 3.4 Karşılaştırma tabanı

Bir tespit ediciye ait "günde 10 alarm" değeri tek başına yorumlanamaz; ancak yerini alacağı yöntemin aynı kayıt üzerindeki başarımına karşı anlam kazanmaktadır. Bu nedenle klasik STA/LTA yöntemi dördüncü bir kol olarak aynı pencerelerde çalıştırılmıştır (STA 0,5 s, LTA 10 s, düşey bileşen, özyinelemeli biçim).

Karakteristik işlev pencere başına değil **sürekli parça üzerinde** hesaplanmaktadır: 10 saniyelik bir LTA 6 saniyelik pencereye sığmamakta, pencere başına hesaplamak klasik yöntemi haksız biçimde zayıflatmaktadır. Her pencere içindeki en yüksek karakteristik işlev değerini almaktadır.

### 3.5 Çalışma noktasının tanımı

Sınama kümesinin 0,5 eşiği sürekli veride anlamsızdır (Bölüm 4.1). Bunun yerine kabul edilebilir **günlük yanlış alarm bütçesi** belirlenmekte ve o bütçeyi veren eşik, ölçülen arka plan dağılımının ilgili yüzdeliğinden okunmaktadır. Arka plan, hiçbir katalog olayının koruma aralığına ([P−10 s, P+60 s]) düşmeyen pencerelerden oluşmaktadır.

Yanlış bildirimler sayılırken alarmlar 60 saniyelik pencerede kümelenmektedir; böylece on ardışık pencereye yayılan tek bir gürültü patlaması on değil bir yanlış bildirim saymaktadır.

### 3.6 İki istasyon uyuşması

Uyuşma kuralı şöyle tanımlanmıştır. Her istasyonun alarmları önce kümelenmekte
(60 s'den yakın alarmlar tek bir bildirim sayılmakta, aksi hâlde on pencere
süren tek bir gürültü patlaması on yanlış pozitif olarak sayılırdı), ardından
bir istasyonun her bildirimi için diğerinin ±*w* saniye içinde bir bildirimi
olup olmadığına bakılmaktadır.

**Kataloglanmış olaylar her iki akıştan da çıkarılmaktadır.** Gerçek bir deprem
iki istasyonda da tanım gereği görülmektedir; akışlarda bırakıldığında her biri
garantili bir uyuşma üretmekte ve ölçülen oran, iki istasyonun gürültüsünün ne
kadar örtüştüğünü değil, sürede kaç deprem olduğunu ölçmektedir. MANT–DEMI
çiftinde bu fark belirleyicidir: ortak sürede günde 11,6 kataloglanmış olay
SGO 3'ü aşmakta, ölçülen uyuşma ise günde 0,425'tir — yani olaylar tek başına
ölçümün tamamını açıklayabilirdi. Bu nedenle bildirimler yalnızca hiçbir
kataloglanmış olayın koruma penceresine düşmeyen pencerelerden sayılmaktadır.

*w* keyfi bir ayar değildir. İki istasyonda P varışları arasındaki fark en büyük
değerini olay ikisini birleştiren doğru üzerinde olduğunda alır ve bu değer
**uzaklık / Vp**'dir: MANT–DEMI için 10,5 s, MANT–GCAM için 24,0 s, DEMI–GCAM
için 32,6 s. Katalog olaylarında ölçülen |ΔP| farkları sırasıyla 0,0–11,2 s,
0,9–23,0 s ve 1,2–27,7 s aralığındadır; seçilen pencereler fiziksel sınırı
kapsamakta, ancak payları yoktur.

**İki istasyon aynı eşiğe değil, aynı bütçeye ayarlanmaktadır.** Arka plan
dağılımları birbirinden çok farklı olduğundan (Bölüm 4.8) ortak bir sayısal eşik
iki istasyonda aynı anlama gelmemektedir.

Ölçülen uyuşma oranı, **iki bağımsız akışın üreteceği oranla**
karşılaştırılmaktadır. Oranları *r*_A ve *r*_B olan iki bağımsız Poisson akışı
için, A'nın bir bildiriminin ±*w* içinde en az bir B bildirimi bulundurma
olasılığı 1 − exp(−*r*_B·2*w*)'dir. (Sıkça kullanılan *r*_A·*r*_B·2*w* çarpımı
çift sayısını verir; burada ölçülen nicelik "en az bir eşleşmesi olan A
bildirimi" olduğundan doğru karşılaştırma üstel biçimdir.)

## 4. Bulgular

### 4.1 Sınama eşiği sürekli veriye aktarılamamaktadır

| Kol | Arka plan ortancası | 0,5 eşiğinde günlük alarm |
|---|---|---|
| 6s | **0,8019** | **12.599** |
| ponly | 0,4306 | 79 |
| pnat | 0,3037 | 45 |
| stalta | 1,49 (KİD) | 13.489 |

Çizelge 2. Arka plan puan dağılımı ve sınama eşiğinin sürekli veride ürettiği alarm sayısı.

6 saniyelik model sürekli **gürültü** üzerinde 0,80 ortanca puan vermekte ve sakin bir istasyon gününün %92,7'sini işaretlemektedir. Sınama kümesinden çıkarılan öngörü günde 257 alarmdı; ölçülen değer 12.599'dur.

Bu modelin kullanılabilir eşik bandı ayrıca son derece dardır: arka planın %50'lik dilimi 0,797, %99,9'luk dilimi 0,839'dur. Eşiğin 0,0024 kadar kaydırılması alarm oranını on kat değiştirmektedir.

### 4.2 Nedeni: genlik boşluğu

Gerçek eğitim **gürültü** pencereleri, tek bir çarpan dışında hiçbir şey değiştirilmeden modele verildiğinde:

| Ölçek | Ortanca standart sapma (istasyon σ) | p>0,5 oranı |
|---|---|---|
| 1,000 | 0,708 | 0,019 |
| 0,300 | 0,212 | 0,183 |
| 0,100 | 0,071 | 0,862 |
| 0,010 | 0,007 | 1,000 |

Çizelge 3. 6 saniyelik modelin genliğe tepkisi (gerçek gürültü pencereleri ölçeklenmiştir).

`P(deprem | genlik)` ilişkisi **U biçimlidir**. Genlik madenciliği negatif sınıfın altına, fizik ise pozitif sınıfın altına bir taban koymaktadır; ~0,1 σ değerinin altında model **hiçbir sınıfa ait eğitim verisi görmemiştir** ve dışdeğerlemesi "deprem" yönünde çıkmaktadır. Sürekli MANT arka planı tam olarak 0,11 σ'da bulunmaktadır.

İki olası açıklama sınanmış ve elenmiştir:

- **Taban σ'nın yanlış hesaplanmış olması.** Eğitim gürültü dosyalarındaki rastgele pencereler de aynı aralığa düşmektedir; GADA (0,049), GELI (0,088) ve ENEZ (0,085) gibi eğitim istasyonları MANT'ın 0,11 değeriyle aynı bölgededir.
- **Genlik madenciliğinin kendisi.** `ponly` madencilikli, `pnat` madencilliksizdir ve her ikisi de tekdüzedir. Kusur, S dalgasını içerdiği için pozitifleri 581 σ'ya kadar uzanan 6 saniyelik pencere tasarımına özgüdür.

### 4.3 AUC ile çalışma noktası ters sıralama vermektedir

| Kol | Olay AUC'si (SGO≥3) | 100 alarm/gün | 10 alarm/gün | 1 alarm/gün |
|---|---|---|---|---|
| stalta | **0,9795** | **0,928** | 0,548 | 0,132 |
| ponly | 0,9622 | 0,903 | 0,627 | 0,183 |
| pnat | 0,9516 | 0,861 | 0,573 | 0,219 |
| 6s | 0,9403 | 0,864 | **0,741** | **0,316** |

Çizelge 4. Olay düzeyinde ayırt etme gücü ile çalışma noktalarındaki duyarlılık.

Klasik STA/LTA yöntemi bütün SGO eşiklerinde AUC bakımından birinci, günde 10 ve 1 alarm bütçelerinde ise sonuncudur. Günde 1 alarmda 6 saniyelik model 2,4 kat daha fazla deprem bulmaktadır.

Nedeni ölçütün tanımındadır: AUC bütün ROC eğrisini bütünlemekte, oysa bir çalışma noktası eğrinin tek bir uç köşesinde yaşamaktadır (burada yanlış pozitif oranı 7,4·10⁻⁴). Bir tespit edici genel sıralamada daha iyi, uç kuyrukta daha kötü ayrışabilmektedir.

| Kol | DP | YN | YP | Kesinlik | Duyarlılık | F1 |
|---|---|---|---|---|---|---|
| 6s | 9.675 | 3.381 | 5.326 | 0,645 | **0,741** | **0,690** |
| ponly | 8.181 | 4.875 | 5.624 | 0,593 | 0,627 | 0,609 |
| pnat | 7.481 | 5.575 | 5.829 | 0,562 | 0,573 | 0,568 |
| stalta | 7.152 | 5.904 | 5.952 | 0,546 | 0,548 | 0,547 |

Çizelge 5. Günde 10 alarm bütçesinde olay düzeyinde karışıklık matrisi (SGO≥3, 13.056 olay). Olay düzeyinde doğru negatif hücresi tanımsızdır: "negatif deprem" diye bir nesne yoktur.

### 4.4 Açıklanamayan alarmlar günlük döngü göstermektedir

| Kol | Gündüz (06–20) | Gece | Oran |
|---|---|---|---|
| 6s | 368/saat | 213/saat | 1,73 |
| ponly | 388/saat | 186/saat | 2,08 |
| pnat | 387/saat | 187/saat | 2,07 |
| stalta | 378/saat | 199/saat | 1,90 |

Çizelge 6. Günde 10 alarm bütçesinde açıklanamayan alarmların yerel saate göre dağılımı; dört kolun tamamında tepe 12:00–15:00 arasındadır.

Bu çalışmadaki yanlış alarm sayıları biçimsel olarak **üst sınırdır**: katalogda karşılığı bulunmayan bir alarm ya gerçek bir yanlış pozitiftir ya da kataloglanmamış bir depremdir. Günlük döngü bu sınırı daraltmaktadır: **depremlerin mesai saatlerinde tepe yapması beklenmez.**

### 4.5 P varışının S dalgasından önce bildirilmesi

Bir tespit, penceresinin tamamı gözlenip puanlanmadan bildirilemez. Bu nedenle **alarm zamanı pencerenin sonudur**; başlangıcın kullanılması modele henüz sahip olmadığı bilgiyi atfetmek olurdu. Ölçülen ortanca bildirim gecikmesi 6 saniyelik kol için P+4,5 s, 3,4 saniyelik kol için P+2,9 s'dir.

S−P farkı yaklaşık *uzaklık*/8,4 saniye olduğundan, S dalgasının geçilebilmesi için **S−P farkının bildirim gecikmesini aşması** gerekmektedir. Buradan çıkan başabaş uzaklıklar sırasıyla ~38 km ve ~24 km'dir.

| Uzaklık aralığı | Ortanca S−P | 6s | ponly | stalta |
|---|---|---|---|---|
| 0 – 25 km | 1,4 s | 0,19 | **0,39** | 0,16 |
| 25 – 50 km | 4,0 s | 0,41 | **0,66** | 0,52 |
| 50 – 100 km | 10,7 s | 0,69 | 0,76 | **0,97** |
| 100 – 200 km | 15,1 s | 0,86 | 0,85 | **0,97** |
| 200 – 500 km | 32,1 s | 0,94 | 0,93 | **0,98** |

Çizelge 7. S dalgasından önce bildirilen olayların oranı (günde 10 alarm, SGO≥3).

Ölçülen başabaş uzaklıklar çizelgedeki %50 geçişleriyle örtüşmektedir. Kısa pencere yaklaşık **14 km'lik ek kullanılabilir menzil** kazandırmaktadır; 25 km içinde ise her iki model de çoğunlukla geç kalmaktadır.

Toplamda 6 saniyelik kol tespit ettiği olayların %69,6'sını, P-dalgası kolu %76,0'sını, klasik yöntem ise %94,3'ünü S'den önce bildirmektedir. Klasik yöntemin yüksek oranı, tespitlerinin uzak olaylara kaymasından kaynaklanmaktadır.

### 4.6 Daha sık pencereleme öncelik sorununu çözmemektedir

Bildirim gecikmesinin ne kadarının **ızgara etkisinden** kaynaklandığı ayrı bir deneyle ölçülmüştür: katalog olaylarının çevresi 0,5 saniyelik adımla yeniden taranmıştır.

Aynı eşikte kazanç belirgindir — 6 saniyelik kolun ortanca bildirimi P+4,5 s'den **P+1,9 s**'ye, S'den önce bildirim oranı %69,6'dan **%90,7**'ye çıkmaktadır. Ancak örtüşen pencereler birim zamanda ~12 kat daha fazla karar ürettiğinden, aynı eşikte yanlış bildirim oranı günde 7,3'ten ~137'ye yükselmektedir. Bütçe sabit tutulduğunda:

| Yapılandırma | Eşik | Tespit oranı | S'den önce | Ortanca bildirim |
|---|---|---|---|---|
| 6s, ayrık pencere | 0,8569 | %74,1 | %69,6 | P+4,5 s |
| 6s, sık pencere | 0,8999 | %38,9 | %88,2 | P+2,2 s |
| stalta, sık pencere | 13,41 | %18,8 | **%97,4** | **P+1,4 s** |

Çizelge 8. Sık pencerelemenin eşit yanlış bildirim bütçesinde (günde ~10) etkisi.

Sık pencereleme sabit bütçede duyarlılığı düşürmektedir: eşik 0,857'den 0,900'e çıkmakta, bu da modelin en yüksek puanı olan 0,903'e çok yaklaşmaktadır. Daha önemlisi, **klasik yöntemin öncelik üstünlüğü sık pencerelemede de sürmektedir.** Bunun nedeni yapısaldır: STA/LTA'nın karakteristik işlevi doğrudan dalga başlangıcına tepki vermekte ve bağlam gerektirmemektedir; sinir ağları ise eğitildikleri pencere geometrisi kadar varış sonrası sinyal beklemektedir.

Bu bölümün sayıları 6 parçalık bir alt kümeye (2.445 olay) dayanmakta ve eşik ayarı yalnızca 0,39 günlük temiz olay öncesi gürültüyle yapılmıştır; kuyruk kestirimi gürültülüdür.

### 4.7 Çalışma noktası seçimi

| Bütçe (yanlış alarm/gün) | Eşik | Tespit oranı | S'den önce | Ortanca bildirim |
|---|---|---|---|---|
| 100 | 0,8350 | %86,4 | %76,3 | P+4,3 s |
| **10** | **0,8569** | **%74,1** | **%69,6** | **P+4,5 s** |
| 1 | 0,8950 | %31,6 | %76,8 | P+4,1 s |
| 0,1 | 0,9014 | %6,4 | %91,7 | P+3,4 s |

Çizelge 9. 6 saniyelik model için bütçe–başarım dengesi (SGO≥3, 13.056 olay).

Günde 100 yanlış alarm — yaklaşık saatte dört — kabul edilebiliyorsa kaydedilmiş depremlerin %86'sı bulunmaktadır. Bütçe günde 10'a indirildiğinde duyarlılık %74'e, günde 1'e indirildiğinde %32'ye gerilemektedir: **bütçenin son bir mertebelik daralması duyarlılığın yarısından fazlasına mal olmaktadır.**

### 4.8 Yanlış alarm oranı istasyona özgüdür

Aynı model, aynı ön işleme ve aynı eşik türetme yöntemiyle üç istasyonda
çalıştırıldığında arka plan dağılımları örtüşmemektedir:

| | MANT (728,3 gün) | DEMI (472,0 gün) | GCAM (172,8 gün) |
|---|---|---|---|
| p50 | **0,8019** | **0,8358** | **0,1205** |
| p90 | 0,8272 | 0,8378 | 0,3594 |
| p99 | 0,8345 | 0,8395 | 0,7543 |
| p99,9 | 0,8472 | 0,8644 | 0,8377 |
| en büyük | 0,9031 | 0,9025 | 0,9022 |
| 0,5 üzeri parça aralığı | ~%92,7 | %93,8–99,8 | **%0,44–17,4** |
| σ_Z (uzun dönem) | 963,4 | **2130,4** | **362,9** |

Çizelge 10. 6 saniyelik modelin arka plan puan dağılımı, üç istasyon.

**Üç istasyondan ikisinde model kullanılamaz durumdadır, birinde değildir.**
MANT ve DEMI'de sessiz kaydın neredeyse tamamı 0,5 eşiğini aşmakta (ortanca
0,80 ve 0,84), GCAM'de ise ortanca pencere 0,12 puan almakta, yani model orada
beklendiği gibi davranmaktadır. Fark ortancada 6,7 kattır ve parça bazında
GCAM'in oranı %0,44 ile %17,4 arasında değişirken diğer ikisi hiçbir parçada
%90'ın altına inmemektedir.

Bu bulgu Bölüm 4.2'deki tanıyı iki bağımsız istasyonda desteklemektedir, ve
sıralama σ ile **tek yönlüdür**: σ_Z 2130 olan DEMI en yüksek, 963 olan MANT
ortada, 363 olan GCAM en düşük arka plan puanını vermektedir. Mekanizma bununla
tutarlıdır — σ ender büyük genlikli olaylarca belirlendiğinden, σ'sı büyük olan
istasyonda tipik pencere kendi σ'sının daha küçük bir kesridir, yani genlik
ekseninde sessizlik boşluğunun daha derinine düşmektedir.

**Bu bir tutarlılık gözlemidir, kanıt değildir.** İstasyonlar arası ham sayım
değerleri cihaz duyarlılığına bağlıdır ve burada cihaz tepkisi giderilmemiştir;
bağı kesinleştirmek için her istasyonun tipik pencere genliğinin kendi σ'suna
oranının ölçülmesi gerekmektedir, ki bu ayrı bir arşiv taraması demektir.

İşletme sonucu bundan bağımsız olarak geçerlidir: **bir istasyonda ölçülen
yanlış alarm oranı başka bir istasyona aktarılamaz.** Eşik, her istasyon için
kendi kaydından türetilmek zorundadır. Bu, önceki sürümde "tek istasyon"
başlığıyla açık bir sınırlılık olarak bırakılmış olan sorunun yanıtıdır; yanıt,
değişkenliğin küçük olmadığıdır.

Kaydedilebilirlik de üç istasyonda farklıdır: SGO≥3 oranı DEMI'de %45,2,
MANT'ta %27,0, GCAM'de %11,7'dir. Günde 10 alarm bütçesinde SGO≥3 olaylarının
bulunma oranı sırasıyla 0,528, 0,741 ve 0,639'dur.

### 4.9 Uyuşmanın kazancı istasyon uzaklığına bağlıdır

Her istasyon **günde 10 alarm** bütçesine ayarlandığında, ortak kayıt sürelerinde
ölçülen değerler (yalnızca kataloglanmamış bildirimler):

| Çift | Uzaklık | Model | A/gün | B/gün | 2/2 uyuşma/gün | Bağımsız olsaydı | **Fazlalık** | Duyarlılık | Uyuşma sayısı |
|---|---|---|---|---|---|---|---|---|---|
| MANT–DEMI | 63 km | P-öncelikli | 7,53 | 8,10 | 0,425 | 0,0148 | **28,8×** | 0,369 | 170 |
| MANT–DEMI | 63 km | 6 s | 6,94 | 7,33 | 0,709 | 0,0123 | **57,5×** | 0,563 | 284 |
| MANT–GCAM | 144 km | P-öncelikli | 8,33 | 8,85 | **0,044** | 0,0408 | **1,1×** | 0,371 | 7 |
| MANT–GCAM | 144 km | 6 s | 6,59 | 8,71 | **0,069** | 0,0318 | **2,2×** | 0,444 | 11 |
| DEMI–GCAM | 196 km | P-öncelikli | 7,73 | 9,17 | 0,139 | 0,0534 | 2,6× | 0,125 | 8 |

Çizelge 11. İki istasyon uyuşması, günde 10 alarm bütçesinde.

**Birbirine yakın iki istasyonun yanlış alarmları örtüşmektedir; uzak olanlarınki
örtüşmemektedir.** 63 km aralıklı çiftte ölçülen uyuşma, bağımsızlık
varsayımının öngördüğünün 29–58 katıdır. 144 km aralıklı çiftte 1,1–2,2 kat,
yani varsayım pratikte geçerlidir. İki tespit edici bu sonucu birbirinden
bağımsız olarak vermektedir; dolayısıyla bulgu tek bir modelin özelliği
değildir.

İşletme sonucu doğrudandır: **uzak çift, aynı duyarlılıkta on kat daha az yanlış
bildirim üretmektedir.** 144 km'de 23 günde bir, 63 km'de 2,4 günde bir yanlış
bildirim. Bunu yalnızca ölçerek görmek mümkündür, çünkü aritmetik tersini
söylemektedir: yakın çiftin uyuşma penceresi daha dardır (±10,5 s'ye karşı
±24,0 s) ve rastlantısal uyuşmayı azaltması beklenir. Örtüşme bu avantajı
tümüyle silmektedir.

**Örneklem büyüklüğü konusunda açık olmak gerekir.** Yukarıdaki son sütun
uyuşma sayısıdır ve üç çift eşit güvenilirlikte değildir:

- MANT–DEMI, 170 ve 284 uyuşma üzerinde ölçülmüştür. Poisson belirsizliği ±13
  ve ±17'dir; 29–58 kat fazlalık güvenle 1'in üzerindedir. **Bu bulgu sağlamdır.**
- MANT–GCAM ve DEMI–GCAM ise 7, 11 ve 8 uyuşma üzerinde ölçülmüştür (±2,6–3,3).
  Bu sayılarla 1 kat ile 3 kat ayırt edilemez. "144 km'de tam bağımsızlık"
  **iddia edilememektedir**; 196 km'nin 144 km'den yüksek çıkması da gürültüdür.

Buna karşılık **oran farkı örneklem etkisi değildir**: uzak çift yakın çift
kadar örtüşseydi GCAM'in 159 gününde yaklaşık 68 uyuşma beklenirdi, 7 ölçülmüştür.

**Fazlalık, eşik sıkılaştıkça büyümektedir** (MANT–DEMI, P-öncelikli): günde 100
alarmda 3,3×, 30'da 9,7×, 10'da 28,8×, 3'te 85,8×. Gevşek eşiklerde neredeyse
her pencere alarm verdiğinden uyuşma rastlantıya yakındır; eşik sıkıldıkça
ayakta kalan uyuşmalar giderek daha çok gerçekten paylaşılan bir sinyalin izini
taşımaktadır. İki olası açıklama vardır ve bu ölçümle ayırt edilememektedir:

1. **Ortak kaynaklı gürültü.** 63 km aralıklı iki istasyon hava koşullarını,
   bölgesel gürültü alanını ve Bölüm 4.4'te görülen kültürel etkinliği büyük
   ölçüde paylaşmaktadır. Bunlar gerçek yanlış alarmlardır ve hiçbir ağ kuralı
   temizleyemez.
2. **Katalogda bulunmayan gerçek depremler.** 63 km aralıklı iki istasyonun
   10 saniye içinde yüksek puanlı bir geçici sinyalde uyuşması, tam olarak
   yakın bir gerçek depremin görünümüdür — ve DEMI üç istasyonun en duyarlısıdır.

Bu nedenle günde 0,425 değeri **yanlış alarm için bir üst sınır**, kaçırılmış
katalog olayları için bir alt sınırdır. Ortak sürede 170 uyuşan bildirim
bulunmaktadır; bu elle incelenebilir bir listedir ve iki açıklamayı ayırmanın
yolu budur.

**Duyarlılık bedeli.** İki istasyon şart koşulduğunda SGO≥3 olaylarının bulunma
oranı MANT–DEMI'de 0,369 (P-öncelikli) ve 0,563 (6 s) olmaktadır. Bu değerler
tek istasyon duyarlılığıyla **doğrudan karşılaştırılamaz**: tek istasyon
duyarlılığı MANT'ta 13.056 olay üzerinde, uyuşma duyarlılığı ise iki istasyonda
birden SGO≥3 koşulunu sağlayan olaylar üzerinde ölçülmektedir (MANT–DEMI için
4.645, diğer çiftler için 48–133).

## 5. Tartışma

**Düzenlenmiş sınama kümesi bir çalışma noktası belirleyememektedir.** 6 saniyelik modelin 0,9896 AUC değeri geçerliliğini korumaktadır; aktarılamayan şey AUC değil, o küme üzerinde belirlenen eşiktir. Bu ayrım önemlidir, çünkü yayımlanmış başarım değerleri genellikle eşikle birlikte raporlanmakta ve eşik sessizce taşınmaktadır.

**Kusur veri kümesi tasarımındadır, mimaride değil.** Negatif örneklerin genliğe göre seçilmesi, modelin eğitim sırasında sessiz pencere görmemesine yol açmaktadır. Zhu ve Beroza (2019) ile Ravishan vd. (2026) aynı sorunu farklı sözcüklerle belirtmiştir. Bu çalışmanın katkısı, sorunun **büyüklüğünü** ölçmektir: sürekli arka plan, eğitim negatiflerinin bulunduğu genlik bandının bir mertebe altındadır.

**Toplulaştırılmış ölçütler işletme kararı için yeterli değildir.** Dört tespit ediciyi AUC ile sıralamak, işletme koşullarındaki sıralamanın tam tersini vermektedir. Bu, bu modellere özgü bir tuhaflık değil, ölçütün tanımından çıkan bir sonuçtur ve herhangi bir karşılaştırma için geçerlidir.

**Öncelik süresi bir mimari sınırdır.** Varış sonrası bağlam gerektiren bir model, doğrudan başlangıca tepki veren bir orana yetişememektedir. İyileştirmenin yolu tarama biçimi değil, varış sonrası kısmı daha kısa olan bir pencereyle yeniden eğitmektir — bu da tespit başarımından ödün vermek anlamına gelmektedir.

**Ağ uyuşmasının kazancı bir geometri sorusudur, aritmetik sorusu değil.**
Yayımlanmış tartışmalarda 2/N uyuşmasının getirisi genellikle bağımsızlık
varsayımı altında hesaplanmaktadır. Bu çalışmada varsayım 144 km'de geçerli,
63 km'de ise 29–58 kat yanlış çıkmaktadır. Yani soru "2/N ne kadar kazandırır"
değil, "**istasyonlar birbirinden ne kadar uzak**" sorusudur; ve aritmetik bu
noktada yanıltıcıdır, çünkü yakın çiftin uyuşma penceresi daha dar olduğundan
kâğıt üzerinde daha iyi görünmektedir.

**Ölçüm iki yönlü bilgi vermektedir.** Uyuşan alarmların bir bölümü kataloglanmamış
deprem ise, aynı ölçüm hem yanlış alarm oranına üst sınır koymakta hem de
katalogun eksikliğine alt sınır koymaktadır. Bölüm 4.4'teki günlük döngü,
MANT'ta tek istasyon alarmlarının önemli bir bölümünün kültürel olduğunu
göstermişti; iki istasyon uyuşması bu ayrımı yapmanın ikinci ve daha doğrudan
yoludur.

### 5.1 Sınırlılıklar

- **Üç istasyon bir ağ değildir.** Bölüm 4.8 ve 4.9 ile önceki sürümdeki "tek istasyon" ve "ilişkilendirme öncesi" sınırlılıkları giderilmiştir; ancak 2/N kuralının N>2 için davranışı ölçülmemiştir.
- **Uzaklık ekseninde yalnızca bir nokta sağlamdır.** MANT–DEMI 170 ve 284 uyuşma üzerinde ölçülmüştür; diğer iki çift 7–11 uyuşma üzerinde. Örtüşmenin hangi uzaklıkta kaybolduğu **belirlenmemiştir**, yalnızca 63 km'de güçlü ve 144 km'de zayıf olduğu gösterilmiştir. Bunu daraltmak için uzun süre eşzamanlı kayıt yapan daha çok istasyon çifti gerekmektedir; GCAM 9 Aralık 2024'te durduğundan bu veriyle daraltılamaz.
- **Uzaklık, istasyon ve mevsimle karışmaktadır.** Üç çiftin ortak süreleri farklı takvim aralıklarını kapsamaktadır (400, 159 ve 58 gün), dolayısıyla "uzaklık" etkisi istasyonların kendi gürültü ortamlarından ve mevsimden tam olarak ayrıştırılmış değildir.
- **Fazlalığın kaynağı ayrıştırılmamıştır.** MANT–DEMI'de ölçülen 29–58 katlık fazlalığın ne kadarının ortak gürültü, ne kadarının kataloglanmamış deprem olduğu belirlenmemiştir; 170 uyuşan bildirimin elle incelenmesi gerekmektedir.
- **Uyuşma pencerelerinin payı yoktur.** Her çift için ±uzaklık/Vp fiziksel sınırdır ve ölçülen en büyük farklar bu sınırın hemen altındadır; istasyon çifti değiştiğinde yeniden hesaplanmalıdır.
- **Gecikme ayrıştırması yapılmamıştır.** Yalnızca model terimi bilinmektedir; telemetri, tamponlama ve karar aktarımı dâhil değildir.
- **Kalibre edilmemiş belirsizlik.** Sinir ağı çıkışları kalibre edilmiş olasılık değildir.
- **Ayarlanmamış karşılaştırma tabanı.** STA/LTA için ders kitabı değerleri (0,5/10 s) kullanılmıştır; ayarlanmış bir klasik seçici daha iyi başarım gösterebilir.

## 6. Sonuçlar

1. Sınama kümesinde belirlenen 0,5 eşiği sürekli veriye aktarılamamaktadır: öngörülen 257'ye karşılık günde 12.599 alarm ölçülmüştür. Eşik, ölçülen arka plan dağılımından türetilmek zorundadır.
2. Nedeni, negatif örneklerin genliğe göre seçilmesinden doğan ve modelin hiç eğitim verisi görmediği bir sessizlik bölgesidir; sürekli arka plan tam olarak orada bulunmaktadır.
3. Olay düzeyinde AUC ile işletme çalışma noktası dört tespit ediciyi ters sırada dizmektedir; işletmeye yönelik hiçbir karşılaştırma yalnızca AUC ile raporlanmamalıdır.
4. Açıklanamayan alarmların günlük döngüsü, önemli bir bölümünün kültürel gürültü olduğunu göstermektedir.
5. S dalgasından önce bildirim yeteneği, pencere boyunun dayattığı bir başabaş uzaklığa bağlıdır (~38 km ve ~24 km); 25 km içinde her iki model de çoğunlukla geç kalmaktadır.
6. Öncelik süresi tarama sıklığıyla iyileştirilememektedir; bu bir ayar sorunu değil mimari bir sınırdır.
7. Yanlış alarm oranı istasyona özgüdür: aynı model MANT'ta 0,80, DEMI'de 0,84, GCAM'de 0,12 arka plan ortancası vermektedir ve sıralama istasyonun uzun dönem σ'suyla tek yönlüdür. Eşik her istasyon için kendi kaydından türetilmelidir.
8. İki istasyonun uyuşmasının kazancı istasyon uzaklığına bağlıdır. 63 km aralıklı çiftte yanlış alarmlar bağımsızlığın öngördüğünün 29–58 katı örtüşmekte, 144 km'de varsayım geçerli olmaktadır. Uzak çift aynı duyarlılıkta on kat daha az yanlış bildirim vermektedir (23 günde bir, 2,4 güne karşılık). Bir ağ tasarlanırken istasyonların birbirine yakınlığı bir avantaj değil, maliyettir.
9. Bu fazlalığın bir bölümü kataloglanmamış deprem olabilir; iki istasyonun 10 saniye içinde uyuşması gerçek bir yakın depremin görünümüdür. Dolayısıyla ölçülen uyuşma oranı yanlış alarm için bir **üst sınır**, katalog eksikliği için bir alt sınırdır.

Bu çalışma, kısa pencereli sınıflandırıcıların sürekli veride kullanılamayacağını göstermemektedir. Gösterdiği şey, **düzenlenmiş bir sınama kümesinde belirlenen çalışma noktasının sürekli veriye taşınamayacağı** ve tespit edicilerin yalnızca toplulaştırılmış bir ölçütle karşılaştırılmasının yanıltıcı olabileceğidir.

## Kaynaklar

Hu, Y., Zhang, Q., Zhao, W., & Wang, H. (2021). TransQuake: A transformer-based deep learning approach for seismic P-wave detection. *Earthquake Research Advances*, 1, 100004.

Ravishan, D., Prasanna, R., Herath, P., & Doyle, E. E. H. (2026). Lightweight convolutional neural network for real-time earthquake P-wave detection on edge devices in New Zealand. *Scientific Reports*.

Zhu, W., & Beroza, G. C. (2019). PhaseNet: A deep-neural-network-based seismic arrival-time picking method. *Geophysical Journal International*, 216(1), 261–273.

---

*Ölçümler `src/sismokaos/continuous/cli.py` ile yapılmıştır (`baseline`, `scan`, `report`, `timing`, `coincidence` alt komutları). Puan dosyaları `scores_mant/` ve `scores_gcam/`, eşik ve uyuşma çizelgeleri `final_*.csv`, `gcam_6s_*.csv` ve `coinc_*_coincidence.csv` altındadır.*
