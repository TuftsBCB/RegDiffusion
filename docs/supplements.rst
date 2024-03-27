Additional local networks
=========================

Here we provide some additional local networks we generated based on the 
Hammond microglia datasets using RegDiffusion. 

.. raw:: html

    <select id="iframeSelector">
        <option value="rd_hammond_Sall1.html">Sall1</option>
        <option value="rd_hammond_Hexb.html">Hexb</option>
        <option value="rd_hammond_Fcrls.html">Fcrls</option>
        <option value="rd_hammond_Cx3cr1.html">Cx3cr1</option>
        <option value="rd_hammond_Tmem119.html">Tmem119</option>
        <option value="rd_hammond_Trem2.html">Trem2</option>
        <option value="rd_hammond_P2ry12.html">P2ry12</option>
        <option value="rd_hammond_Mertk.html">Mertk</option>
        <option value="rd_hammond_Pros1.html">Pros1</option>
        <option value="rd_hammond_Siglech.html">Siglech</option>
    </select>

    <iframe id="iframeDisplay" src="rd_hammond_Sall1.html" width="100%" height="500"></iframe>

    <script type="text/javascript">
        document.getElementById('iframeSelector').addEventListener('change', function() {
            var selectedPage = this.value;
            document.getElementById('iframeDisplay').src = selectedPage;
        });
    </script>