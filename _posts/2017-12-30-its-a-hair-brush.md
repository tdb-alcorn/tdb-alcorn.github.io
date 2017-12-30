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
    </head>
    <style>
    </style>
    <div id='canvas'>
    </div>
    <script type='text/javascript'>
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
            this.maxLength = random(3,15);
            this.hairColor = round(random(0, 255));
            // this.wiggliness = random(0.4);
            // this.growiness = random(1, 1.2);
            // this.maxLength = random(20,80);
            // this.hairColor = round(random(80, 255));
            function draw() {
                var h = this.end.subtract(this.start);
                var l = h.rotate(this.f * PI/2);
                var r = h.rotate(-this.f * PI/2);
                var m0 = l.scale(this.p0).add(h.scale(1./3.)).add(this.start);
                var m1 = r.scale(this.p1).add(h.scale(-1./3.)).add(this.end);
                stroke(this.hairColor, 120, 255);
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
        var timestep = 0.05;
        var hairs = [];
        var center;
        function setup() {
            randomSeed(42);
            var canvas = createCanvas(600, 400);
            canvas.parent('canvas');
            stroke(255);
            fill(0, 0);
            background(0);
            strokeWeight(3);
            strokeCap(ROUND);
            strokeJoin(ROUND);
            colorMode(HSB, 255);
            center = new Vector(width/2.0, height/2.0);
        }
        function draw() {
            background(0);
            for (var i=0, len=hairs.length; i<len; i++) {
                hairs[i].draw();
            }
            for (var i=0, len=hairs.length; i<len; i++) {
                hairs[i].update(timestep);
            }
            if (mouseIsPressed) {
                var r = new Vector(mouseX, mouseY);
                hairs.push(new Hair(r, r.add(((new Vector(10, 0)).rotate(random(2*PI))))));
            }
        }
    </script>
</div>