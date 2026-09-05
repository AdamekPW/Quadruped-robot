""" Sensory - czym robot mierzy świat """

from mjlab.scene import SceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)

from robodog.assets.robots.silver_badger.constants import TERRAIN_SCAN_SITE_NAME

from ..constants import (
    FEET_CONTACT_SENSOR_NAME,
    FOOT_GEOMS,
    FOOT_HEIGHT_SCAN_SENSOR_NAME,
    FOOT_SITES,
    TERRAIN_SCAN_SENSOR_NAME,
    TRUNK_CONTACT_SENSOR_NAME,
)

# Promień i liczba rzutów pierścienia pod stopą.
_FOOT_RING_RADIUS = 0.04
_FOOT_RING_SAMPLES = 4


def prepare_sensors(scene: SceneCfg) -> None:
    """Przestraja sensory bazy i dokłada dwa sensory kontaktu."""
    for sensor in scene.sensors or ():
        if sensor.name == TERRAIN_SCAN_SENSOR_NAME:
            assert isinstance(sensor, RayCastSensorCfg)
            assert isinstance(sensor.frame, ObjRef)
            # NIE tułów: promienie lecą pionowo w dół z fizycznej pozycji ramki,
            # więc startując z tułowia wchodzą pod powierzchnię zbocza stromszego
            # niż ~22° i raportują ścianę jako płaską podłogę.
            sensor.frame.type = "site"
            sensor.frame.name = TERRAIN_SCAN_SITE_NAME
        if sensor.name == FOOT_HEIGHT_SCAN_SENSOR_NAME:
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITES
            )
            sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

    feet_ground_cfg = ContactSensorCfg(
        name=FEET_CONTACT_SENSOR_NAME,
        primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    trunk_ground_cfg = ContactSensorCfg(
        name=TRUNK_CONTACT_SENSOR_NAME,
        primary=ContactMatch(
            mode="geom", pattern=("trunk_collision", "rear_collision"), entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    scene.sensors = (scene.sensors or ()) + (
        feet_ground_cfg,
        trunk_ground_cfg,
    )
