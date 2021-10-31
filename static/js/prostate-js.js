function fillTheData()
{
	psaLow = 2;
	psaHigh = 50;
	psaStep = 1;
	var options = '';
	for(var i = psaLow; i <= psaHigh; i+=psaStep)
		options += '<option value="'+ parseFloat(i).toFixed(2) +'" />';
	document.getElementById('psaList').innerHTML = options;
	
	ageLow = 40;
	ageHigh = 90;
	ageStep = 1;	
	options = '';
	for(var i = ageLow; i <= ageHigh; i+=ageStep)
		options += '<option value="'+ parseInt(i) +'" />';
	document.getElementById('ageList').innerHTML = options;	
	

	options = '';
	options += '<option value="25" />';
	options += '<option value="40" />';
	options += '<option value="60" />';
	document.getElementById('pvList').innerHTML = options;		
}