async function getPreferences(){
    const response = await fetch('http://localhost:8080/get_preferences', {
                                method: 'GET',
                            });
    let preferences = await response.json();

    document.getElementById("ocr-language").value = preferences.ocr.lang;
    document.getElementById("ocr-reescale").value = preferences.ocr.reescale_percent;

    document.getElementById("table-nms_threshold").value = preferences.tablas.nms_threshold;
    document.getElementById("table-score_threshold").value = preferences.tablas.score_threshold;
    document.getElementById("table-offset").value = preferences.tablas.offset;

    document.getElementById("cardinality-nms_threshold").value = preferences.cardinalidades.nms_threshold;
    document.getElementById("cardinality-score_threshold").value = preferences.cardinalidades.score_threshold;
    document.getElementById("cardinality-distance_threshold").value = preferences.cardinalidades.distance_threshold;
}


document.getElementById("savePreferencesButton").addEventListener("click", async function() {
    const preferences = {"tablas":{}, "cardinalidades":{}, "ocr":{}}
    
    preferences['ocr']['lang'] = document.getElementById("ocr-language").value;
    preferences['ocr']['reescale_percent'] = parseFloat(document.getElementById("ocr-reescale").value);
    preferences['tablas']['nms_threshold'] = parseFloat(document.getElementById("table-nms_threshold").value);
    preferences['tablas']['score_threshold'] = parseFloat(document.getElementById("table-score_threshold").value);
    preferences['tablas']['offset'] = parseFloat(document.getElementById("table-offset").value);

    preferences['cardinalidades']['nms_threshold'] = parseFloat(document.getElementById("cardinality-nms_threshold").value);
    preferences['cardinalidades']['score_threshold'] = parseFloat(document.getElementById("cardinality-score_threshold").value);
    preferences['cardinalidades']['distance_threshold'] = parseFloat(document.getElementById("cardinality-distance_threshold").value);

    await fetch('http://localhost:8080/update_preferences', {
                        method: 'POST',
                        body: JSON.stringify(preferences),
                        headers: {
                            'Accept': 'application/json',
                            'Content-Type': 'application/json'
                            },
                    });
});

getPreferences()