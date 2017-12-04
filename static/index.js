(function() {
	//crete a new canvas object of type 2d with the Width and Height both at 280px
	var canvas = document.querySelector("#canvas");
	var context = canvas.getContext("2d");
	canvas.width = 280;
	canvas.height = 280;

	//create a mouse and set its default x and y to 0,0
	var Mouse = {x:0, y:0};
	var lastMouse = {x:0, y:0};
	//set the canvas fill color to white
	context.fillStyle = "white";
	//cover the fill to the width and height specified
	context.fillRect(0, 0, canvas.width, canvas.height);
	//set the color of the line to black
	context.color = "black";
	context.lineWidth = 10;
    context.lineJoin = context.lineCap = 'round';
	
	debug();

	canvas.addEventListener("mousemove", function(e) {
		lastMouse.x = Mouse.x;
		lastMouse.y = Mouse.y;

		Mouse.x = e.pageX - this.offsetLeft 1;
		Mouse.y = e.pageY - this.offsetTop 1;
	}, false);

	canvas.addEventListener("mousedown", function(e) {
		canvas.addEventListener("mousemove", onPaint, false);
	}, false);

	canvas.addEventListener("mouseup", function() {
		canvas.removeEventListener("mousemove", onPaint, false);
	}, false);

	var onPaint = function() {	
		context.lineWidth = context.lineWidth;
		context.lineJoin = "round";
		context.lineCap = "round";
		context.strokeStyle = context.color;
	
		context.beginPath();
		context.moveTo(lastMouse.x, lastMouse.y);
		context.lineTo(Mouse.x,Mouse.y );
		context.closePath();
		context.stroke();
	};

	function debug() {
		$("#clearButton").on("click", function() {
			context.clearRect( 0, 0, 280, 280 );
			context.fillStyle="white";
			context.fillRect(0,0,canvas.width,canvas.height);
		});
	}
}());