<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<%@ taglib prefix="c"   uri="http://java.sun.com/jsp/jstl/core" %>
<%@taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>뉴스 텍스트 분석 결과 차트</title>
    <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
    <script type="text/javascript">
      google.charts.load('current', {'packages':['corechart']});
      google.charts.setOnLoadCallback(drawVisualization);

      var category = new Array();
	  var all_data = new Array();
	  var correct = new Array();
	  var model = new Array();
	  var rate = new Array();

      <c:forEach items="${list }" var="TextAnalysisModel">
		  category.push("${TextAnalysisModel.category }");
		  all_data.push("${TextAnalysisModel.all_data }");
		  correct.push("${TextAnalysisModel.correct_data }");
		  rate.push(100/"${TextAnalysisModel.all_data }"*"${TextAnalysisModel.correct_data }")
		  model.push("${TextAnalysisModel.model}")
	  </c:forEach>
	  
      function drawVisualization() {
        // Some raw data (not necessarily accurate)
        var data = google.visualization.arrayToDataTable([
          ['Category', 'RNN', { role: 'annotation' }, 'LSTM',{ role: 'annotation' }, { role: 'style' }, 'BiLSTM', { role: 'annotation' }],
          [category[0], Number(rate[6]),'RNN', Number(rate[3]), 'LSTM', 'green', Number(rate[0]),'BiLSTM'],
          [category[1], Number(rate[7]),'RNN', Number(rate[4]), 'LSTM', 'green', Number(rate[1]), 'BiLSTM'],
          [category[2], Number(rate[8]),'RNN', Number(rate[5]), 'LSTM', 'green', Number(rate[2]), 'BiLSTM']
        ]);

        var options = {
          title : 'Predictive success rate by model',
          vAxis: {title: 'Rate'},
          hAxis: {title: 'Category'},
          seriesType: 'bars',
          series: {1: { color: 'green' },
              		3: {type: 'line'}}
        };

        var chart = new google.visualization.ComboChart(document.getElementById('chart_div'));
        chart.draw(data, options);
      }
    </script>
</head>
<body>
	<div id="chart_div" style="width: 900px; height: 500px;"></div>
	<%-- <h1>훈련 결과</h1>
    <br>
	<c:forEach items="${list }" begin="1" end="1" var="TextAnalysisModel">
		<h2>모델 정확도 : ${TextAnalysisModel.evaluate*100 }%</h2>
	</c:forEach>
	<div id="columnchart_material" style="width: 800px; height: 500px;"></div>
	
	<c:set var="all_data" value="0" />
	<c:set var="correct_data" value="0" />
	<c:forEach items="${list }" var="TextAnalysisModel">
		  <h2>${TextAnalysisModel.category } 카테고리 정확도: 
		  <c:set var="accuracy" value="${100/TextAnalysisModel.all_data*TextAnalysisModel.correct_data}"/>
		  <c:out value="${accuracy}"/>%</h2>
		  <c:set var="all_data" value="${all_data + TextAnalysisModel.all_data}" />
		  <c:set var="correct_data" value="${correct_data + TextAnalysisModel.correct_data}" />
	</c:forEach>
	<h3>전체 데이터 개수: <c:out value="${all_data }"/>개</h3>
	<h3>맞은 데이터 개수: <c:out value="${correct_data }"/>개</h3> --%>
</body>
</html>