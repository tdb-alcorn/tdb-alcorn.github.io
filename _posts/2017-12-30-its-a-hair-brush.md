---
layout: post
date: 2017-12-29 18:10:53 -0800
image: /assets/img/small-wreck-tom.jpg
title: It's A Hair Brush
excerpt: Wherein I appeal to your inner kindergartener
---

<div id='p5-container'>
    <head>
        <script type='text/javascript' src='https://cdnjs.cloudflare.com/ajax/libs/p5.js/0.5.16/p5.min.js'></script>
        <script type='text/javascript' src='{{ "/assets/lib/vector.js" | absolute_url }}'></script>
        <script type='text/javascript' src='{{ "/assets/lib/hairsDraw.js" | absolute_url }}'></script>
    </head>
    <style>
        #colorPointPreview {
            width: 20px;
            height: 20px;
            background-color: hsl(180, 50%, 50%);
            display: inline-block;
        }
    </style>
    <div id='canvas'>
    </div>
    <div id='controls'>
        <form>
            <label>Color:</label><input id='colorPoint' type='range' value='180' min='0' max='359' oninput='updateControls();'><div id='colorPointPreview'></div>
            <label>Color variance:</label><input id='colorVariance' type='range' value='30' min='0' max='180' oninput='updateControls();'>
            <label>Max length:</label><input id='globalMaxLength' type='range' value='15' min='1' max='50' oninput='updateControls();'>
        </form>
        <button onclick='reset();'>Reset</button>
        <button onclick='saveToFile();'>Save</button>
        <button onclick='showFilePicker();'>Load</button>
        <input id='filePicker' type="file" style='display:none;' oninput='loadFromFile(); hideFilePicker();'>
    </div>
    <script type='text/javascript'>
        var colorPoint = 180;
        var colorVariance = 30;
        var globalMaxLength = 15;
        function updateControls() {
            colorPoint = parseInt(document.getElementById('colorPoint').value);
            document.getElementById('colorPointPreview').style.backgroundColor = 'hsl(' + colorPoint + ', 50%, 50%)';
            colorVariance = parseInt(document.getElementById('colorVariance').value);
            globalMaxLength = parseInt(document.getElementById('globalMaxLength').value);
        }
        function Hair(start, end, p0, p1) {
            this.start = start;
            this.end = end;
            this.p0 = p0 || random(0.5);
            this.p1 = p1 || random(0.5);
            this.f = 1;
            if (random(1) > 0.5) {
                this.f = -1;
            }
            this.wiggliness = random(0.5, 1.0);
            this.growiness = random(1, 1.10);
            this.maxLength = random(3, globalMaxLength);
            this.hairColor = round(colorPoint + random(-colorVariance, colorVariance)) % 360;
            function draw() {
                var h = this.end.subtract(this.start);
                var l = h.rotate(this.f * PI/2);
                var r = h.rotate(-this.f * PI/2);
                var m0 = l.scale(this.p0).add(h.scale(1./3.)).add(this.start);
                var m1 = r.scale(this.p1).add(h.scale(-1./3.)).add(this.end);
                stroke(this.hairColor, 50, 100);
                bezier(this.start.x, this.start.y, m0.x, m0.y, m1.x, m1.y, this.end.x, this.end.y);
            }
            function update(timestep) {
                this.p0 = constrain(this.p0 + this.wiggliness * random(-timestep, timestep), 0, 1);
                this.p1 = constrain(this.p1 + this.wiggliness * random(-timestep, timestep), 0, 1);
                var h = this.end.subtract(this.start);
                if (h.lengthSq() < (this.maxLength*this.maxLength)) {
                    this.end = this.start.add(h.scale(this.growiness));
                }
            }
            this.draw = draw;
            this.update = update;
        }
        Hair.rehydrate = function(obj) {
            var h = Object.assign(new Hair(), obj);
            h.start = Vector.rehydrate(h.start);
            h.end = Vector.rehydrate(h.end);
            return h;
        }
        var timestep = 0.05;
        var hairs = [];
        var center;
        function reset() {
            hairs = [];
        }
        function showFilePicker() {
            document.getElementById('filePicker').style.display = 'inline';
        }
        function hideFilePicker() {
            document.getElementById('filePicker').style.display = 'none';
        }
        function loadFromFile() {
            var file = document.getElementById('filePicker').files[0];
            var reader = new FileReader();
            reader.onload = function(e) {
                loadFromJSON(e.target.result);
            };
            reader.readAsText(file);
        }
        function loadFromJSON(json) {
            hairs = JSON.parse(json);
            for (var i=0, len=hairs.length; i<len; i++) {
                hairs[i] = Hair.rehydrate(hairs[i]);
            }
        }
        function saveToFile() {
            saveJSON(hairs, 'hairs.json');
        }
        function setup() {
            var parentWidth = document.getElementById('canvas').offsetWidth;
            randomSeed(42);
            var canvas = createCanvas(parentWidth, Math.round(parentWidth*0.618));
            canvas.parent('canvas');
            stroke(255);
            fill(0, 0);
            background(0);
            strokeWeight(3);
            strokeCap(ROUND);
            strokeJoin(ROUND);
            colorMode(HSB);
            center = new Vector(width/2.0, height/2.0);
            loadFromJSON(hairsDrawJson);
        }
        function draw() {
            background(0);
            for (var i=0, len=hairs.length; i<len; i++) {
                hairs[i].draw();
            }
            for (var i=0, len=hairs.length; i<len; i++) {
                hairs[i].update(timestep);
            }
        }
        function mouseDragged() {
            var r = new Vector(mouseX, mouseY);
            // Only draw hairs if mouse is on canvas.
            if (r.within(0, 0, width, height)) {
                hairs.push(new Hair(r, r.add(((new Vector(10, 0)).rotate(random(2*PI))))));
            }
            return true;
        }
    </script>
</div>