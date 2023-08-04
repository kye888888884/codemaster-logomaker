const btn = document.getElementById("btn");
const input_text = document.getElementById("text");

btn.addEventListener("click", () => {
    let csrftoken = document.getElementsByName("csrfmiddlewaretoken")[0].value;

    let data = {
        text: input_text.value,
    };

    $.ajax({
        type: "POST",
        url: "/generate/",
        headers: { "X-CSRFToken": csrftoken },
        cache: false,
        data: JSON.stringify(data),
    }).done(function (data) {
        data_image = data["image"];
        // console.log(data_image);

        // .result-container 초기화
        let result_container = document.getElementById("result-container");
        result_container.innerHTML = "";

        // 이미지 태그 추가
        let img = document.createElement("img");
        img.src = data_image;
        img.width = 512;
        img.height = 512;
        img.alt = "logo";
        img.id = "result";
        // .result-container에 이미지 태그 추가
        result_container.appendChild(img);
    });
});
