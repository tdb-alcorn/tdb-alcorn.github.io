---
layout: post
date:   2017-12-30 20:59:53 -0800
image: /assets/img/small-wreck-tom.jpg
title: 'Seeing Like A Perceptron II: Training'
excerpt: It's maths, not magic
---

<div id='container'>
    <head>
        <link rel='stylesheet' src='https://cdn.pydata.org/bokeh/release/bokeh-0.12.9.min.css'>
        <script type='text/javascript' src='https://cdn.pydata.org/bokeh/release/bokeh-0.12.9.min.js'></script>
        <script type='text/javascript' src='https://cdn.pydata.org/bokeh/release/bokeh-api-0.12.9.min.js'></script>
        <link rel='stylesheet' src='https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css'>
        <script type='text/javascript' src='https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js'></script>
        <script type='text/javascript' src='{{ "/assets/lib/perceptrons.js" | absolute_url }}'></script>
    </head>
    <style>
        #graph {
            display: flex;
            flex-direction: row;
            justify-content: center;
        }
    </style>
    <label>Epoch</label><input id='epoch' type='range' min='0' max='999' value='200' step='1' oninput='updatePlot();'><label id='epochValue'>200</label>
    <div id='graph'>
        <div id='transformedDataPlot'></div>
        <div id='trainingAccuracyPlot'></div>
    </div>
    <script type='text/javascript'>
        perceptron = perceptronGenerator(relu);
        var epoch = 200;
        var data = new vis.DataSet();
        data.add({x:0,y:0,z:0,style:0});
        // var accData = new vis.DataSet();
        // accData.add({x:0,y:0,z:0,style:0});
        var options = {
            width:  '600px',
            height: '600px',
            style: 'dot-color',
            showPerspective: true,
            showGrid: true,
            showLegend: false,
            keepAspectRatio: true,
            verticalRatio: 1.0,
            cameraPosition: {
                horizontal: -0.35,
                vertical: 0.22,
                distance: 1.8
            },
        };
        // var accOptions = {
        //     width: '300px',
        //     height: '300px',
        //     // style: 'line',
        //     legend: false,
        //     sort: false,
        //     dataAxis: {
        //         left: {
        //             range: {
        //                 min: 300, max: 800
        //             }
        //         }
        //     },
        //     // showGrid: true,
        //     // verticalRatio: 1.0,
        // };
        // var accGroups = new vis.DataSet();
        // accGroups.add({
        //     id: 0,
        //     options: {
        //         style: 'line',
        //     },
        // });
        // accGroups.add({
        //     id: 1,
        //     // content: names[3],
        //     options: {
        //         style: 'points',
        //         // drawPoints: {
        //         //     style: 'circle' // square, circle
        //         // },
        //         // shaded: {
        //         //     orientation: 'top' // top, bottom
        //         // }
        //     },
        // });
        var graph = new vis.Graph3d(document.getElementById('transformedDataPlot'), data, options);
        // var accGraph = new vis.Graph2d(document.getElementById('trainingAccuracyPlot'), accData, accGroups, accOptions);
        var updatePlot;
        fetch('{{ "/assets/data/data_0.json" | absolute_url }}').then((response) => response.json())
        .then((data0) => {
            fetch('{{ "/assets/data/data_1.json" | absolute_url }}').then((response) => response.json())
            .then((data1) => {
                fetch('{{ "/assets/data/weights.json" | absolute_url }}').then((response) => response.json())
                .then((weights) => {
                    fetch('{{ "/assets/data/accuracy.json" | absolute_url }}').then((response) => response.json())
                    .then((accuracy) => {
                        accuracy = accuracy.map((a, i) => {
                            return {x: i, y: a, group: 0};
                        });
                        plot(data0, data1, weights, accuracy, epoch);
                        updatePlot = function() {
                            epoch = parseInt(document.getElementById('epoch').value);
                            document.getElementById('epochValue').innerText = epoch;
                            plot(data0, data1, weights, accuracy, epoch);
                        }
                    });
                });
            });
        });
        function plot(data0, data1, weights, accuracy, epoch) {
            var d = new vis.DataSet();
            var perceptrons = weights[epoch];
            var dt0 = transform(perceptrons, data0);
            var dt1 = transform(perceptrons, data1);
            dt0.map((x) => d.add({x:x[0], y:x[1], z:x[2], style:0}));
            dt1.map((x) => d.add({x:x[0], y:x[1], z:x[2], style:1}));
            graph.setData(d);
            // var accD = new vis.DataSet(accuracy.concat([{x: epoch, y: accuracy[epoch].y, group: 1}]));         
            // accGraph.setItems(accD);
        }
    </script>
</div>