import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

collision_energies = {
    "Orbitrap": 25,
    "cfm-predict 4": 40,
    "None": 40,
    "ESI-QFT": 30,
    "LC-ESI-QTOF": 30,
    "QTOF": 30,
    "LC-ESI-ITFT": 30,
    "ESI-QTOF": 30,
    "qTof": 30,
    "LC-ESI-QFT": 30,
    "Hybrid FT": 25,
    "LC-ESI-QIT": 20,
    "QQQ": 30,
    "qToF": 30,
    "LC-Q-TOF/MS": 30,
    "Maxis II HD Q-TOF Bruker": 30,
    "LC-APPI-QQ": 30,
    "Quattro_QQQ:25eV": 25,
    "Ion Trap": 20,
    "Q-Exactive Plus Orbitrap Res 70k": 25,
    "LC-ESI-ITTOF": 30,
    "Sciex Triple TOF 6600": 30,
    "impact HD": 30,
    "FAB-EBEB": 25,
    "Maxis HD qTOF": 30,
    "LC-QTOF": 30,
    "quadrupole-orbitrap": 25,
    "LC-ESI-Orbitrap": 25,
    "Q-Exactive Plus": 25,
    "Quattro_QQQ:40eV": 40,
    "ESI-TOF": 30,
    "LC-ESI-TOF": 30,
    "LC-APCI-ITFT": 30,
    "ESI-FT": 30,
    "LC-ESI-QQ": 30,
    "Linear Ion Trap": 25,
    "APCI-ITFT": 30,
    "LC-ESI-QEHF": 30,
    "qTOF": 30,
    "GC-APCI-QTOF": 30,
    "ESI-ITFT": 30,
    "Q-Exactive Plus Orbitrap Res 14k": 25,
    "QIT": 20,
    "SCIEX TripleTOF 6600": 30,
    "Q-TOF": 30,
    "Agilent 6530 QTOF": 30,
    "Quattro_QQQ:10eV": 10,
    "LC-ESI-Q-Orbitrap": 25,
    "MALDI-TOFTOF": 25,
    "SYNAPT QTOF, Waters": 30,
    "Hybrid Ft": 25,
    "Thermo Q Exactive HF": 25,
    "Agilent 6530 Q-TOF": 30,
    "Q-TOF SYNAPT, Waters": 30,
    "MALDI-QITTOF": 25,
}

# Updated CollisionEnergyConfig to classify HCD, CID, and mz-specific values


@dataclass
class CollisionEnergyConfigWithClassification:
    default: Union[int, str] = 20  # Default energy if none specified
    hierarchy: Dict[str, Union[int, str]] = field(
        default_factory=lambda: collision_energies
    )

    def extract_numeric_energy(
        self, energy_str: Optional[str], machine_type: Optional[str]
    ) -> float:
        """Extract and normalize numeric energy, including classification for HCD, CID, and mz-specific formats."""
        if not energy_str:
            return float(self.hierarchy.get(machine_type, self.default))

        # Handle ramp cases by averaging values in the range
        if "Ramp" in energy_str or "-" in energy_str:
            range_match = re.findall(r"(\d+(\.\d+)?)", energy_str)
            if len(range_match) >= 2:
                values = [float(v[0]) for v in range_match]
                return sum(values) / len(values)

        # Handle percentage cases (HCD, NCE) by referencing the nominal value from the machine type
        if "%" in energy_str or "NCE" in energy_str:
            # Extract percentage value and apply it to the machine-specific nominal
            percent_match = re.search(r"(\d+(\.\d+)?)", energy_str)
            if percent_match and machine_type:
                nominal_value = self.hierarchy.get(machine_type, self.default)
                percent_value = float(percent_match.group(1)) / 100.0
                return percent_value * nominal_value

        # CID values - directly interpret unless specific instruction required
        if "CID" in energy_str:
            numeric_match = re.search(r"(\d+(\.\d+)?)", energy_str)
            if numeric_match:
                return float(numeric_match.group(1))

        # Handle mz-specific formats, focusing on the primary energy value
        if "mz" in energy_str:
            # Assume the main value is the first numeric value, ignoring mz adjustments
            main_value_match = re.search(r"(\d+(\.\d+)?)", energy_str)
            if main_value_match:
                return float(main_value_match.group(1))

        # Default to a direct numeric extraction if possible
        numeric_match = re.search(r"(\d+(\.\d+)?)", energy_str)
        if numeric_match:
            return float(numeric_match.group(1))

        return float(self.default)

    def prepare_for_embedding(
        self, energy_str: Optional[str], machine_type: Optional[str]
    ) -> float:
        """Prepare the collision energy for embedding by extracting normalized values with classification."""
        return self.extract_numeric_energy(energy_str, machine_type)
