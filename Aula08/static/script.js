// Aguarda a página carregar
function retorno_ajax(resp){
    console.log(resp);

    new Chart($('#chart'), {
        type: 'bar',
        data: {
          labels: ['Notícia sobre esporte', 'Notícia qualquer'],
          datasets: [
            {
              backgroundColor: ['#3e95cd', '#3cba9f'],
              data: [0.95, 0.05]
            }
          ]
        },
        options: {
          legend: { display: false },
          title: {
            display: true,
            text: 'Probabilidade da notícia ser sobre esporte'
          }
        }
    });
}


$(document).ready(function(){

    $('#txt_area').on('change keyup paste', function() {
        
        // Obtém o conteúdo
        var txt = $(this).val();

        // Envia o conteúdo via AJAX
        $.ajax({
            type: 'POST',
            url: '/analisa_noticia',
            data: {'txt': txt},
            success: retorno_ajax
        });

        // console.log(txt);
    });
});