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

      <c:forEach items="${list }" var="TextAnalysisModel">
		  category.push("${TextAnalysisModel.category }");
		  all_data.push("${TextAnalysisModel.all_data }");
		  correct.push("${TextAnalysisModel.correct_data }");
		  model.push("${TextAnalysisModel.model}")
	  </c:forEach>
	  
      function drawVisualization() {
        // Some raw data (not necessarily accurate)
        var data = google.visualization.arrayToDataTable([
          ['Category', 'LSTM',{ role: 'annotation' }, 'BiLSTM', { role: 'annotation' }, { role: 'style' }, 'RNN', { role: 'annotation' }, 'All Data'],
          [category[0],  Number(correct[3]), 'LSTM', Number(correct[0]),'BiLSTM', 'gold', Number(correct[6]),'RNN', Number(all_data[0])],
          [category[1],  Number(correct[4]), 'LSTM', Number(correct[1]), 'BiLSTM', 'gold', Number(correct[7]),'RNN',Number(all_data[1])],
          [category[2],  Number(correct[5]), 'LSTM', Number(correct[2]), 'BiLSTM', 'gold', Number(correct[8]),'RNN', Number(all_data[2])]
        ]);

        var options = {
          title : 'Number of matched data in total',
          vAxis: {title: 'Number of data'},
          hAxis: {title: 'Category'},
          seriesType: 'bars',
          series: {1: { color: 'gold' },
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