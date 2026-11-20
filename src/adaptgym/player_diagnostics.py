"""Deterministic saved-player annotations; never calls models or changes profiles."""
from adaptgym.claim_metadata_metrics import _claims, _kind, score_error_diagnostics, summarize_error_diagnostics
from adaptgym.verticals import VerticalSchema


def annotate_evaluation(evaluation, schema):
    vertical = VerticalSchema(schema)
    reports = []
    for checkpoint in evaluation['checkpoints']:
        profile, reference = checkpoint.get('profile'), checkpoint['reference_profile']
        # Prior profiles can retain citations outside the three-message input window.
        user_ids = checkpoint.get('reference_user_message_ids')
        if user_ids is None:
            user_ids = [m['id'] for m in checkpoint.get('messages', []) if m.get('role') == 'user']
        report = score_error_diagnostics(profile, reference, vertical, user_ids)
        checkpoint['error_diagnostics'] = report
        checkpoint['claim_kinds'] = {
            name: [_kind(c, vertical) for c in _claims(value)]
            for name, value in [('prediction', profile), ('reference', reference)]}
        reports.append(report)
    evaluation['error_summary'] = summarize_error_diagnostics(reports)
    return evaluation


def annotate_bundle(bundle):
    schema = bundle['sample']['plan']['vertical_schema']
    for modes in bundle['models'].values():
        for evaluation in modes.values():
            annotate_evaluation(evaluation, schema)
    return bundle
