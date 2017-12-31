function loadJSON(f) {
    return fetch(f).then((response) => {
        return response.json();
    });
}