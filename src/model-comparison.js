var manavgat = ee.Geometry.Point([31.48, 36.88]); 
Map.centerObject(manavgat, 12); 

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");
var yanginSonrasi = s2.filterBounds(manavgat)
                     .filterDate('2021-08-10', '2021-08-25') 
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5)) 
                     .median();

Map.addLayer(yanginSonrasi, {bands: ['B12', 'B8', 'B4'], min: 0, max: 5000}, 'Uydu Görüntüsü');

var yanmisEtiketli = yanmis.map(function(f) { return f.set('class', 1); });
var saglikliEtiketli = saglikli.map(function(f) { return f.set('class', 0); });
var egitimVerisi = yanmisEtiketli.merge(saglikliEtiketli);

var bantlar = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];

var testPikselleri = yanginSonrasi.select(bantlar).sampleRegions({
  collection: egitimVerisi,
  properties: ['class'],
  scale: 30,
  geometries: true
}).randomColumn()
  .filter(ee.Filter.lt('random', 0.1)); 
  
print('SVM için hafifletilmiş piksel sayısı:', testPikselleri.size());

print('Poligonların içinden okunan toplam piksel sayısı:', testPikselleri.size());
print('İlk 5 pikselin verisi (Görmek için):', testPikselleri.limit(5));

var veriSeti = testPikselleri.randomColumn();
var egitimSeti = veriSeti.filter(ee.Filter.lt('random', 0.7));
var testSeti = veriSeti.filter(ee.Filter.gte('random', 0.7));

var rf = ee.Classifier.smileRandomForest(100).train({
  features: egitimSeti,
  classProperty: 'class',
  inputProperties: bantlar
});

var svm = ee.Classifier.libsvm().train({
  features: egitimSeti,
  classProperty: 'class',
  inputProperties: bantlar
});

var rfTest = testSeti.classify(rf);
var rfAcc = rfTest.errorMatrix('class', 'classification').accuracy();

var svmTest = testSeti.classify(svm);
var svmAcc = svmTest.errorMatrix('class', 'classification').accuracy();

print('--- SONUÇLAR GELDİ ---');
print('Random Forest Doğruluğu (RF):', rfAcc);
print('SVM Doğruluğu:', svmAcc);

var rfHarita = yanginSonrasi.select(bantlar).classify(rf);
var svmHarita = yanginSonrasi.select(bantlar).classify(svm);

Map.addLayer(rfHarita, {min:0, max:1, palette: ['white', 'red']}, 'RF Sonucu (Kırmızı)');
Map.addLayer(svmHarita, {min:0, max:1, palette: ['white', 'blue']}, 'SVM Sonucu (Mavi)');


var rfStats = rfHarita.eq(1).multiply(ee.Image.pixelArea()).reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: manavgat.buffer(20000).bounds(), 
  scale: 30,
  maxPixels: 1e13
});

var svmStats = svmHarita.eq(1).multiply(ee.Image.pixelArea()).reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: manavgat.buffer(20000).bounds(),
  scale: 30,
  maxPixels: 1e13
});

var rfHa = ee.Number(rfStats.get('classification')).divide(10000);
var svmHa = ee.Number(svmStats.get('classification')).divide(10000);

print('--- ALAN ANALİZİ ---');
print('Random Forest (RF) Toplam Yanan Alan (ha):', rfHa);
print('SVM Toplam Yanan Alan (ha):', svmHa);
print('Modeller Arasındaki Fark (ha):', rfHa.subtract(svmHa).abs());
