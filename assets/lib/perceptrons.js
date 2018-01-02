function sigmoid(x) {
    return 1 / (1 + Math.exp(-1 * x));
}

function relu(x) {
    return (x + Math.abs(x)) / 2;
}

function perceptron(weights, x) {
    return sigmoid(dot(weights, x));
}

function perceptronGenerator(activation) {
    return function(weights, x) {
        return activation(dot(weights, x));
    }
}

function dot(x, y) {
    if (x.length !== y.length) {
        throw new Error("Length mismatch");
    }
    return x.map((_, i) => x[i] * y[i]).reduce((acc, val) => acc + val, 0);
}

function transform(perceptrons, inputs) {
    return inputs.map((x) => perceptrons.map((w) => perceptron(w, x)));
}

function range(n) {
    var r = Array(n);
    for (let i=0; i<n; i++) {
        r[i] = i;
    }
    return r;
}

function linspace(min, max, num) {
    var step = (max-min)/(num-1);
    return range(num).map((s) => min + s * step);
}

function first(pair) {
    return pair[0];
}

function second(pair) {
    return pair[1];
}

function third(tuple) {
    return tuple[2];
}