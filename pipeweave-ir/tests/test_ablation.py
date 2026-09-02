from pipeweave_ir.ablation import deterministic_group_split


def test_deterministic_group_split_keeps_groups_disjoint() -> None:
    rows = [
        {"problem_group_id": f"shape-{group}", "row": row} for group in range(5) for row in range(3)
    ]
    train, test, metadata = deterministic_group_split(
        rows,
        group_field="problem_group_id",
        test_fraction=0.4,
        seed=17,
    )
    train_groups = {row["problem_group_id"] for row in train}
    test_groups = {row["problem_group_id"] for row in test}
    assert train_groups.isdisjoint(test_groups)
    assert metadata["train_rows"] == len(train)
    assert metadata["test_rows"] == len(test)


def test_deterministic_group_split_is_stable() -> None:
    rows = [{"config_group_id": f"config-{index}"} for index in range(8)]
    first = deterministic_group_split(
        rows, group_field="config_group_id", test_fraction=0.25, seed=9
    )
    second = deterministic_group_split(
        list(reversed(rows)), group_field="config_group_id", test_fraction=0.25, seed=9
    )
    assert first[2]["test_groups"] == second[2]["test_groups"]
