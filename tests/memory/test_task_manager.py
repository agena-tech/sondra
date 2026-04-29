from sondra.memory.task_manager import _fallback_task_state


def test_fallback_task_state_detects_su_an_variant() -> None:
    result = _fallback_task_state("Şu an memory debug yapıyorum.")

    assert result == {"goal": "", "step": "memory debug"}


def test_fallback_task_state_detects_current_step_variant() -> None:
    result = _fallback_task_state("Şu anki adımım normalize testleri yapmak.")

    assert result == {"goal": "", "step": "normalize testleri yapmak"}


def test_fallback_task_state_detects_kontrol_ediyorum_variant() -> None:
    result = _fallback_task_state("Şu anda correction hattını kontrol ediyorum.")

    assert result == {"goal": "", "step": "correction hattını"}
