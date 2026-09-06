# Pilot Çalışma: Kademeli Sistemin S Dalgasından Önce Sonuç Üretme Oranı

**Taslak — 06.09.2026.** Bu belge bir pilot çalışmanın ön bulgularını
sunmaktadır; sonuçlar tek istasyona ve tek tespit koluna dayanmakta olup
yayımlanabilir nitelikte değildir. Sınırlamalar Bölüm 6'da açıkça
belirtilmiştir.

---

## 1. Özet

Derin öğrenmeye dayalı bir deprem tespit edicisinin S dalgasından önce alarm
üretebildiği daha önce ölçülmüştü (%69,6). Ancak bir erken uyarı sisteminin
üretmesi gereken çıktı yalnızca "deprem var" bilgisi değil, **büyüklük
kestirimi** ile birliktedir. Bu çalışma, **tespit ve büyüklük kestiriminin her
ikisinin birden** S dalgası varmadan önce tamamlanma oranını ölçmektedir.

Ana bulgu şudur: **belirleyici kısıt tespit edici değil, büyüklük penceresinin
uzunluğudur.** Tespit tek başına olayların %69,5'inde S'den önce
gerçekleşmektedir. Buna 6 saniyelik bir büyüklük penceresi eklendiğinde oran
%62,7'ye (yaklaşık 7 puan kayıp), 10 saniyelik pencerede %45,0'e, 20 saniyelik
pencerede ise %4,6'ya düşmektedir.

Bu, doğrudan bir **mühendislik ödünleşimi** ortaya koymaktadır: daha uzun
büyüklük penceresi daha iyi büyüklük kestirimi vermekte (6 s → 10 s geçişi
ortalama mutlak hatayı iyileştirmektedir), fakat aynı uzatma erken uyarı
yeteneğini ortadan kaldırmaktadır.

---

## 2. Amaç

Projenin önceki raporlarında tespit ediciler için S dalgasından önce bildirim
oranları verilmiştir. Bu oranlar **yalnızca birinci kademeyi** tanımlamaktadır.
Kademeli sistemde ikinci kademe, kendi penceresi dolmadan çıktı üretememektedir.
Dolayısıyla işletme açısından anlamlı soru şudur:

> Hangi olaylarda hem tespit hem de büyüklük kestirimi, S dalgası istasyona
> ulaşmadan önce hazır olmaktadır?

---

## 3. Yöntem

### 3.1 Zamanlama aritmetiği

`alarm_epoch` değeri, tespit penceresinin **sonudur**. Bir tespit edici, pencere
tamamlanmadan karar üretemeyeceği için alarm zamanı pencerenin başlangıcı değil
bitişidir. Bu ayrım önemlidir: başlangıç zamanının kullanılması, modele henüz
sahip olmadığı bilgiyi atfetmek olurdu.

İkinci kademe, alarm üzerine çapalanmış `W` saniyelik bir pencereye ihtiyaç
duymaktadır. `--anchor-lag L` ve `--pre P` parametreleriyle bu pencere
`alarm_epoch − L − P` anında başlamakta ve

```
mag_hazır = alarm_epoch − L − P + W
```

anında tamamlanmaktadır. Buna göre kademeli sistem, S dalgasını yalnızca

```
dt_vs_s + (W − P − L) < 0        (dt_vs_s = alarm_epoch − s_epoch)
```

koşulu sağlandığında geçmektedir. `dt_vs_s` değeri `sk falsealarm timing`
çıktısında hâlihazırda bulunduğundan, bu ölçüm **yeni bir eğitim veya yeniden
tarama gerektirmemektedir**.

### 3.2 Başabaş uzaklık

S−P farkı uzaklıkla büyümekte, buna karşılık kademeli sistemin kendi gecikmesi
sabit kalmaktadır. Dolayısıyla belirli bir uzaklığın ötesinde sistem her zaman
kazanmakta, altında ise her zaman kaybetmektedir. Bu **başabaş uzaklık**, modelin
değil **pencere geometrisinin** bir özelliğidir.

Bu çalışmada başabaş uzaklık varsayılmamış, ölçülmüştür: tespit edilen olayların
yarısından fazlasının S'den önce sonuçlandığı ilk uzaklık bandı olarak
tanımlanmıştır. S−P ile uzaklık arasındaki ilişki de bir Vp değeri
varsayılmadan, verinin kendisinden uyarlanmıştır.

### 3.3 Veri ve komut

Ölçüm, MANT istasyonunun 728 günlük kesintisiz kaydı üzerinde 6 saniyelik
tespit koluna ait `timing` çıktısına dayanmaktadır.

```bash
python -m experiments.analyses.cascade_lead_time \
    --timing mant_alarm_times.csv --mag-window 6 10 20 --pre 2.0
```

---

## 4. Bulgular

### 4.1 Genel

| | değer |
|---|---|
| Dosyadaki olay sayısı | 48.434 |
| Tespit edilen olay | 11.188 (%23,1) |
| Tespit tek başına S'den önce | 7.780 (%69,5) |
| Ölçülen S−P eğimi | 100 km başına 10,77 s (eşdeğer 9,28 km/s), n = 11.188 |

Tespit tek başına elde edilen %69,5 oranı, `surekli_veri_yanlis_alarm_raporu`
Bölüm 4.5'te bağımsız olarak hesaplanan %69,6 değeriyle örtüşmektedir; bu, iki
hesabın birbirini doğruladığı anlamına gelmektedir.

### 4.2 Büyüklük penceresinin bedeli

**Çizelge 1.** Kademeli sistemin S dalgasını geçme oranı, büyüklük penceresi
uzunluğuna göre.

| Büyüklük penceresi | Eklenen gecikme | Kademeli sistem S'den önce | Başabaş uzaklık |
|---|---|---|---|
| 6 s | +4,0 s | **%62,7** | 50 km |
| 10 s | +8,0 s | %45,0 | 100 km |
| 20 s | +18,0 s | %4,6 | 200 km |

Tespit tek başına %69,5'tir. Buna göre 6 saniyelik büyüklük penceresi yaklaşık
**7 puan**, 10 saniyelik pencere **24,5 puan**, 20 saniyelik pencere ise
**64,9 puan** kayba yol açmaktadır.

### 4.3 Uzaklığa göre dağılım

**Çizelge 2.** Tespit edilen olaylar içinde kademeli sistemin S'den önce
sonuçlandığı oran.

| Uzaklık (km) | Olay | Yalnız tespit | W = 6 s | W = 10 s | W = 20 s |
|---|---|---|---|---|---|
| 0 – 25 | 125 | %41,6 | %0,8 | %0,0 | %0,0 |
| 25 – 50 | 252 | %66,7 | %6,7 | %0,4 | %0,0 |
| 50 – 100 | 9.353 | %68,5 | **%62,9** | %42,1 | %0,6 |
| 100 – 150 | 654 | %81,0 | **%79,2** | **%77,7** | %0,9 |
| 150 – 200 | 245 | %72,2 | %70,6 | %69,8 | %27,8 |
| 200 – 300 | 308 | %72,4 | %69,2 | %67,5 | %61,4 |
| 300 – 500 | 251 | %87,3 | %83,7 | %82,1 | %75,7 |

Üç gözlem öne çıkmaktadır.

**Kör bölge gerçektir ve ölçülmüştür.** 0–25 km bandında kademeli sistem
olayların yalnızca %0,8'inde (6 s penceresi) S'den önce sonuçlanmaktadır; 10
saniyelik pencerede hiçbirinde. Bu, erken uyarı yazınında bilinen kör bölge
olgusudur ve burada varsayılmadan ölçülmüştür.

**100–150 km bandında pencere uzunluğu neredeyse bedelsizdir.** 6 s ile 10 s
arasındaki fark bu bantta yalnızca 1,5 puandır (%79,2'ye karşı %77,7). Bu bant,
daha uzun pencerenin büyüklük doğruluğu kazancının **bedelsiz** elde edildiği
aralıktır.

**Veri kütlesi 50–100 km bandındadır.** Tespit edilen olayların %83,6'sı
(9.353 / 11.188) bu banttadır ve tam da pencere uzunluğunun en pahalı olduğu
yerdir: %62,9'a karşı %42,1. Genel oranlar büyük ölçüde bu bandın davranışını
yansıtmaktadır.

---

## 5. Tartışma

Bulgular, büyüklük kestirimi ile erken uyarı arasında **ölçülmüş bir ödünleşim
eğrisi** ortaya koymaktadır. Proje kayıtlarında büyüklük penceresinin
uzatılmasının kestirim hatasını iyileştirdiği (6 s → 10 s) belgelenmiştir. Bu
çalışma, aynı uzatmanın erken uyarı yeteneğinde 24,5 puanlık bir kayba mal
olduğunu göstermektedir.

Bu iki bulgu birlikte okunduğunda, pencere uzunluğunun **tek bir "en iyi" değeri
bulunmadığı**, seçimin uygulama amacına bağlı olduğu anlaşılmaktadır:

- **Erken uyarı amaçlı** bir kullanımda 6 saniyelik pencere tercih edilmelidir;
  tespit tek başına elde edilen orana yaklaşık 7 puanlık bir maliyetle
  ulaşmaktadır.
- **Hızlı büyüklük bildirimi** (kataloglama, otomatik raporlama) amaçlı bir
  kullanımda 10 saniyelik pencere daha iyi kestirim vermekte, S'den önce
  yetişme koşulu ise aranmamaktadır.
- **20 saniyelik pencere erken uyarı için kullanılamaz** niteliktedir ve proje
  kayıtlarında büyüklük doğruluğu açısından da 10 saniyelik pencereye üstünlük
  sağlamamaktadır.

Ayrıca, **başabaş uzaklığın pencere geometrisinin bir özelliği olduğu**
vurgulanmalıdır. 6 s için 50 km, 10 s için 100 km, 20 s için 200 km olarak
ölçülen bu değerler modelin başarımından değil, doğrudan pencere uzunluğundan
kaynaklanmaktadır. Bir ağ tasarımında istasyon aralığı bu değerlerle birlikte
düşünülmelidir.

---

## 6. Sınırlamalar

Bu bölüm, bulguların hangi koşullarda geçerli olmadığını belirtmektedir.

**Alarm zamanları pencere adımına göre nicemlenmiştir.** 6 saniyelik kolun adımı
6 saniye olduğundan, `dt_vs_s` değeri ±6 s çözünürlüktedir. S−P farkının
yaklaşık 6 saniyenin altında kaldığı **0–50 km bandı bu çözünürlüğün altındadır**
ve bu bandın sayıları yön göstericidir, kesin değildir. Bu bantlar için sık
pencereleme ile yeniden tarama (`scan --near-csv`) gerekmektedir.

**Tek istasyon, tek tespit kolu.** Ölçüm yalnızca MANT istasyonunda ve yalnızca
6 saniyelik kolda yapılmıştır. P-dalgası kolunun S'den önce bildirim oranı daha
yüksektir (%76,0) ve klasik STA/LTA yöntemi bu ölçütte her ikisini de
geçmektedir (%94,3); dolayısıyla kol seçimi sonucu değiştirecektir.

**Büyüklük kestiriminin doğruluğu bu çalışmada ölçülmemiştir.** Burada yalnızca
kestirimin **ne zaman hazır olduğu** hesaplanmıştır, ne kadar doğru olduğu değil.
Kestirim hatası büyüklük bandına güçlü biçimde bağlıdır (M ≤ 2,5 için 0,1430;
M > 3 için 0,4085) ve erken uyarı kararının duyarlı olduğu bant ikincisidir.

**Yalnızca tespit edilen olaylar üzerinde hesaplanmıştır.** Katalogdaki 48.434
olayın %23,1'i tespit edilmiştir; geri kalanı için zamanlama tanımsızdır. Bu
oran, MANT'ta kataloglanmış olayların yalnızca %27,5'inin SGO 3 eşiğine
ulaşmasıyla tutarlıdır.

**Ölçülen S−P eğimi (100 km başına 10,77 s) episantr uzaklığına göredir.**
Odak derinliği hesaba katılmadığından, sığ olmayan olaylarda gerçek hiposantr
uzaklığı daha büyüktür.

---

## 7. Sonuç ve öneriler

1. Kademeli sistem, tespit edilen olayların **%62,7'sinde** (6 s büyüklük
   penceresi ile) hem tespiti hem büyüklük kestirimini S dalgası varmadan
   tamamlamaktadır.
2. Belirleyici kısıt **tespit edici değil, büyüklük penceresinin uzunluğudur**;
   tespit tek başına %69,5'e ulaşmaktadır.
3. Başabaş uzaklık pencere geometrisiyle belirlenmektedir: 6 s için 50 km,
   10 s için 100 km, 20 s için 200 km.
4. 0–25 km kör bölgesinde kademeli sistem işlevsizdir; bu beklenen bir sonuçtur
   ve burada ölçülmüştür.

**Önerilen sonraki adımlar.** (i) 0–50 km bandının sık pencereleme ile yeniden
taranması, çünkü mevcut çözünürlük bu bant için yetersizdir. (ii) Aynı
çözümlemenin P-dalgası kolu ve klasik yöntem için yinelenmesi. (iii) Büyüklük
kestirim hatasının, S'den önce yetişen olaylar alt kümesinde ayrıca
raporlanması — erken uyarı için anlamlı olan doğruluk budur.

---

*Ölçüm `experiments/analyses/cascade_lead_time.py` ile yapılmıştır; girdi
`mant_alarm_times.csv` (`sk falsealarm timing` çıktısı). Çözümleme yeni bir
eğitim veya tarama gerektirmemiştir.*
