const btn = document.getElementById("btn");
const btnClear = document.getElementById("clear");
const canvas = document.getElementById("canvas");
const canvas28 = document.getElementById("canvas28");

const ctx = canvas.getContext("2d");

let drawing = false;
var preX = -1;
var preY = -1;

const draw = (e) => {
    const x = e.offsetX;
    const y = e.offsetY;
    if (!drawing) return;
    ctx.lineWidth = 30;
    ctx.lineCap = "round";
    ctx.strokeStyle = "#000000";
    ctx.beginPath();
    if (preX < 0) {
        ctx.moveTo(x, y);
    } else {
        ctx.moveTo(preX, preY);
    }
    ctx.lineTo(x, y);
    ctx.stroke();
    preX = x;
    preY = y;
};

const init = (force = false) => {
    // 흰 사각형으로 초기화
    if (force) {
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    drawing = false;
    preX = -1;
    preY = -1;
};

init(true);

canvas.addEventListener("mousedown", (e) => {
    drawing = true;
    draw(e);
});
canvas.addEventListener("mouseup", () => {
    init();
    ctx.beginPath();
});
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseout", () => {
    init();
    ctx.beginPath();
});

btnClear.addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
});

btn.addEventListener("click", () => {
    // canvas에서 이미지를 가져와 28x28로 축소하여 canvas28에 그린다.
    const imgBase64 = canvas.toDataURL();
    var ctx28 = canvas28.getContext("2d");
    const img = new Image();
    img.src = imgBase64;
    img.onload = function () {
        ctx28.fillStyle = "#ffffff";
        ctx28.fillRect(0, 0, canvas28.width, canvas28.height);
        ctx28.drawImage(img, 0, 0, 28, 28);

        // canvas28에서 이미지를 가져와 base64로 인코딩한다.
        const imgBase64_28 = canvas28.toDataURL();

        let formData = new FormData();
        formData.append("image_data", imgBase64_28);

        let csrftoken = document.getElementsByName("csrfmiddlewaretoken")[0]
            .value;

        $.ajax({
            type: "POST",
            url: "/mnist/",
            headers: { "X-CSRFToken": csrftoken },
            cache: false,
            data: formData,
            processData: false,
            contentType: false,
        }).done(function (data) {
            // console.log(data);
            for (let i = 1; i <= 3; i++) {
                let tag_idx = "#result-index" + i;
                let tag_val = "#result-value" + i;
                let index = data[i - 1]["index"];
                let value = data[i - 1]["value"];
                value = value * 100;
                value = value.toFixed(2);
                if (value < 0.01) {
                    index = "-";
                }
                $(tag_idx).text(index);
                $(tag_val).text(`(${value}%)`);
            }
        });
    };
});
