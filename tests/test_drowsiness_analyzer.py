from ai.intelligence.drowsiness_analyzer import DrowsinessAnalyzer


def run_test(
    name,
    perclos,
    prolonged_closure,
    yawn_active,
    prolonged_downward
):
    analyzer = DrowsinessAnalyzer()

    temporal_data = {
        "prolonged_closure": prolonged_closure
    }

    yawn_data = {
        "yawn_active": yawn_active
    }

    head_pose_data = {
        "prolonged_downward": prolonged_downward
    }

    result = analyzer.analyze(
        perclos=perclos,
        temporal_data=temporal_data,
        yawn_data=yawn_data,
        head_pose_data=head_pose_data
    )

    print("=" * 60)
    print(name)
    print("=" * 60)

    print(f"PERCLOS              : {perclos}%")
    print(f"Prolonged Eye Closure: {prolonged_closure}")
    print(f"Yawn Active          : {yawn_active}")
    print(f"Head Down            : {prolonged_downward}")
    print()
    print(f"Drowsiness Score     : {result['score']}")
    print(f"Status               : {result['status']}")

    if result["reasons"]:
        print("Reasons:")

        for reason in result["reasons"]:
            print(f"  - {reason}")

    else:
        print("Reasons              : None")

    print()


def main():

    print("\n")
    print("DROWSINESS INTELLIGENCE ENGINE TEST")
    print("\n")

    # ---------------------------------------------------------
    # TEST 1 — NORMAL / ALERT
    # ---------------------------------------------------------

    run_test(
        name="TEST 1 - Normal Driver",
        perclos=5.0,
        prolonged_closure=False,
        yawn_active=False,
        prolonged_downward=False
    )

    # ---------------------------------------------------------
    # TEST 2 — CAUTION
    # ---------------------------------------------------------

    run_test(
        name="TEST 2 - Elevated PERCLOS",
        perclos=25.0,
        prolonged_closure=False,
        yawn_active=False,
        prolonged_downward=False
    )

    # ---------------------------------------------------------
    # TEST 3 — DROWSY
    # ---------------------------------------------------------

    run_test(
        name="TEST 3 - Prolonged Eye Closure",
        perclos=25.0,
        prolonged_closure=True,
        yawn_active=False,
        prolonged_downward=False
    )

    # ---------------------------------------------------------
    # TEST 4 — MULTIPLE INDICATORS
    # ---------------------------------------------------------

    run_test(
        name="TEST 4 - Multiple Drowsiness Indicators",
        perclos=35.0,
        prolonged_closure=True,
        yawn_active=True,
        prolonged_downward=False
    )

    # ---------------------------------------------------------
    # TEST 5 — CRITICAL
    # ---------------------------------------------------------

    run_test(
        name="TEST 5 - Critical Drowsiness",
        perclos=45.0,
        prolonged_closure=True,
        yawn_active=True,
        prolonged_downward=True
    )


if __name__ == "__main__":
    main()