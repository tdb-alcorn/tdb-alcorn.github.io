function Vector(x, y) {
    this.x = x;
    this.y = y;

    function distanceTo(other) {
      return other.subtract(this).length();
    }
    
    function length() {
      return Math.pow(x*x + y*y, 0.5);
    }
    
    function lengthSq() {
        return x*x + y*y;
    }
    
    function subtract(other) {
      return new Vector(x - other.x, y - other.y);
    }
    
    function add(other) {
      return new Vector(other.x + x, other.y + y);
    }
    
    function scale(c) {
      return new Vector(c * x, c * y);
    }
    
    function setLength(l) {
      return this.scale(l/this.length());
    }
    
    function toString() {
      return "<" + this.x + ", " + this.y + ">";
    }
    
    function rotate(theta) {
      var c = Math.cos(theta);
      var s = Math.sin(theta);
      return new Vector(c * this.x - s * this.y, s * this.x + c * this.y);
    }

    this.distanceTo = distanceTo;
    this.length = length;
    this.lengthSq = lengthSq;
    this.subtract = subtract;
    this.add = add;
    this.scale = scale;
    this.setLength = setLength;
    this.toString = toString;
    this.rotate = rotate;
}