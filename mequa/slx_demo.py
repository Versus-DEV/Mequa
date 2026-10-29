import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def _block(parent, kind, name, sid, params):
    b = ET.SubElement(parent, "Block", BlockType=kind, Name=name, SID=sid)
    for key, value in params.items():
        p = ET.SubElement(b, "P", Name=key)
        p.text = value
    return b


def _line(parent, src, dsts):
    line = ET.SubElement(parent, "Line")
    ET.SubElement(line, "P", Name="ZOrder").text = "1"
    ET.SubElement(line, "P", Name="Src").text = src
    if len(dsts) == 1:
        ET.SubElement(line, "P", Name="Dst").text = dsts[0]
    else:
        for d in dsts:
            br = ET.SubElement(line, "Branch")
            ET.SubElement(br, "P", Name="Dst").text = d
    return line


def make_demo_slx(path):
    system = ET.Element("System")
    ET.SubElement(system, "P", Name="Name").text = "windkessel_demo"
    _block(system, "Sin", "Inflow", "1", {"Amplitude": "4", "Bias": "5", "Frequency": "2*pi*1.2", "Phase": "0"})
    _block(system, "Sum", "Balance", "2", {"Inputs": "|+-"})
    _block(system, "Gain", "InvC", "3", {"Gain": "1/C"})
    _block(system, "Integrator", "Pressure state", "4", {"InitialCondition": "5"})
    _block(system, "Gain", "InvR", "5", {"Gain": "1/R"})
    _block(system, "Outport", "Pressure", "6", {})
    _line(system, "1#out:1", ["2#in:1"])
    _line(system, "2#out:1", ["3#in:1"])
    _line(system, "3#out:1", ["4#in:1"])
    _line(system, "4#out:1", ["5#in:1", "6#in:1"])
    _line(system, "5#out:1", ["2#in:2"])
    diagram = ET.Element("ModelInformation", Version="3.0")
    model = ET.SubElement(diagram, "Model")
    ET.SubElement(model, "P", Name="Name").text = "windkessel_demo"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("simulink/blockdiagram.xml", ET.tostring(diagram, encoding="unicode"))
        z.writestr("simulink/systems/system_root.xml", ET.tostring(system, encoding="unicode"))
    return str(path)


DEMO_VARIABLES = {"R": 1.0, "C": 0.3}
