---
title: Perceptron Visualisation with Javascript
---

<script type="text/javascript" src="https://cdn.plot.ly/plotly-1.31.2.min.js"></script>

<div>
    <ul>
        <li>
            <label>w00</label>
            <input id="w00" type="range" min="-10" max="10" value="10" step="0.1" oninput="updatePerceptrons(); updatePlot();">
        </li>
        <li>
            <label>w01</label>
            <input id="w01" type="range" min="-10" max="10" value="-1" step="0.1" oninput="updatePerceptrons(); updatePlot();">
        </li>
        <li>
            <label>w10</label>
            <input id="w10" type="range" min="-10" max="10" value="-1" step="0.1" oninput="updatePerceptrons(); updatePlot();">
        </li>
        <li>
            <label>w11</label>
            <input id="w11" type="range" min="-10" max="10" value="10" step="0.1" oninput="updatePerceptrons(); updatePlot();">
        </li>
    </ul>
    <div id="plot"></div>
</div>

<script type="text/javascript">
    function sigmoid(x) {
        return 1 / (1 + Math.exp(-1 * x));
    }

    function perceptron(weights, x) {
        return sigmoid(dot(weights, x));
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
        var step = (max - min) / (num - 1);
        return range(num).map((s) => min + s * step);
    }

    function first(pair) {
        return pair[0];
    }

    function second(pair) {
        return pair[1];
    }

    var deepBlue = "#040273";
    var rust = "#a83c09";

    var numGridlines = 100;

    var xValues = linspace(-2.5, 2.5, numGridlines);
    var yValues = linspace(-1.5, 1.5, numGridlines);

    var perceptrons = [
        [10, -1],
        [-1, 10],
    ];

    var data1 = [
        [ 2.29249034,  0.48881005],
        [ 0.71026699,  1.05553444],
        [ 0.05407310,  0.25795342],
        [ 0.58828165,  0.88524424],
        [ 0.29349415,  0.10895031],
        [ 0.03172679,  1.27263986],
        [ 1.07144790,  0.41581801],
    ];
    var data0 = [
        [-0.60754770, -0.12613641],
        [-0.68460636,  0.92871475],
        [-1.84440103, -0.46700242],
        [-1.01700702, -0.13369303],
        [-0.43818550,  0.49344349],
        [-0.19900912, -1.27498361],
        [ 1.55067923, -0.31137892],
        [-1.37923991,  1.37140879],
        [ 0.02771165, -0.32039958],
        [-0.84617041, -0.43342892],
        [-1.33703450,  0.20917217],
        [-1.42432130, -0.55347685],
        [ 0.07479864, -0.50561983],
    ];

    function generateTraces(perceptrons, data0, data1, xValues, yValues) {
        var gridlinesX = yValues.map((y) => xValues.map((x) => [x, y]))
            .map((gx) => transform(perceptrons, gx))
            .map((gx) => {
                return {
                    x: gx.map(first),
                    y: gx.map(second),
                    mode: 'lines',
                    line: {
                        color: deepBlue,
                        width: 1,
                    },
                };
            });

        var gridlinesY = xValues.map((x) => yValues.map((y) => [x, y]))
            .map((gy) => transform(perceptrons, gy))
            .map((gy) => {
                return {
                    x: gy.map(first),
                    y: gy.map(second),
                    mode: 'lines',
                    line: {
                        color: rust,
                        width: 1,
                    },
                };
            });

        var td0 = transform(perceptrons, data0);
        var td1 = transform(perceptrons, data1);

        return [{
            x: td0.map(first),
            y: td0.map(second),
            mode: 'markers',
            marker: {
                color: "#0000ff",
                size: 10,
            },
        },
        {
            x: td1.map(first),
            y: td1.map(second),
            mode: 'markers',
            marker: {
                color: "#ff0000",
                size: 10,
            },
        }]
        .concat(gridlinesX).concat(gridlinesY);
    }

    var layout = {
        showlegend: false,
        width: 500,
        height: 500,
        xaxis: {
            showgrid: false,
        },
        yaxis: {
            showgrid: false,
        },
        margin: {
            t:20, r:20, b:20, l:20,
            pad: 0,
        },
    };

    var options = {
        displayModeBar: false,
        staticPlot: true,
    };

    Plotly.newPlot('plot', generateTraces(perceptrons, data0, data1, xValues, yValues), layout, options);

    function updatePerceptrons() {
        perceptrons[0][0] = document.getElementById("w00").value;
        perceptrons[0][1] = document.getElementById("w01").value;
        perceptrons[1][0] = document.getElementById("w10").value;
        perceptrons[1][1] = document.getElementById("w11").value;
    }

    function updatePlot() {
        var numTraces = xValues.length + yValues.length + 2; // 2 extra ones for the data scatter traces
        Plotly.deleteTraces('plot', range(numTraces));
        Plotly.addTraces('plot', generateTraces(perceptrons, data0, data1, xValues, yValues));
    }
</script>