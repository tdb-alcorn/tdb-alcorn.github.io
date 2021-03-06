function Vector(x, y) {
    this.x = x;
    this.y = y;

    function distanceTo(other) {
      return other.subtract(this).length();
    }
    
    function length() {
      return Math.pow(this.x*this.x + this.y*this.y, 0.5);
    }
    
    function lengthSq() {
        return this.x*this.x + this.y*this.y;
    }
    
    function subtract(other) {
      return new Vector(this.x - other.x, this.y - other.y);
    }
    
    function add(other) {
      return new Vector(other.x + this.x, other.y + this.y);
    }
    
    function scale(c) {
      return new Vector(c * this.x, c * this.y);
    }
    
    function setLength(l) {
      return this.scale(l*l/this.lengthSq());
    }
    
    function toString() {
      return "<" + this.x + ", " + this.y + ">";
    }
    
    function rotate(theta) {
      var c = Math.cos(theta);
      var s = Math.sin(theta);
      return new Vector(c * this.x - s * this.y, s * this.x + c * this.y);
    }
    
    function within(x0, y0, x1, y1) {
        // Returns true if vector is contained within bounds, otherwise false.
        return this.x >= x0 && this.x <= x1 && this.y >= y0 && this.y <= y1;
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
    this.within = within;
}
Vector.rehydrate = function(obj) {
    return Object.assign(new Vector(), obj); 
}