"""Clicks and hover must land on the card that is visually on top
(Code/PLAYTEST_FIX_PLAN.md M1-M5).

The ground truth is built the same way the bug was found: paint every card
in the scene's real draw order into an ID map (each card's actual rendered
pixels, rotation included), then compare with what `Board.card_at` says at
each point.
"""

import unittest

import tests.helpers  # noqa: F401

import pygame

from gui import settings as S
from gui.app import App
from gui.board import draw_key
from gui.engine_bridge import MatchSettings
from gui.scenes.game_scene import GameScene
from gui.sprites.card_sprite import CardSprite

from keyforge.actions import EndTurn, PlayCard
from keyforge.enums import DecisionKind


def _id_map(board):
    surf = pygame.Surface((S.CANVAS_W, S.CANVAS_H))
    surf.fill((0, 0, 0))
    lookup = {}
    for n, cs in enumerate(sorted(board.snapshot.cards.values(), key=draw_key), start=1):
        sp = board.sprites.get(cs.iid)
        if sp is None or not sp.visible or sp.alpha <= 1:
            continue
        face = sp._face_surface()
        t = pygame.transform.rotozoom(face, sp.rot, sp.scale) if (sp.rot or sp.scale != 1.0) else face
        color = (n % 256, n // 256, 9)
        lookup[color] = cs.iid
        m = pygame.mask.from_surface(t, 127).to_surface(setcolor=color, unsetcolor=(0, 0, 0, 0))
        m.set_colorkey((0, 0, 0))
        surf.blit(m, t.get_rect(center=(sp.x, sp.y)))
    return surf, lookup


def _advance_to_busy_board(seed):
    """A human-vs-bot game advanced to a CHOOSE_ACTION with several hand
    cards and creatures on the board."""
    app = App(window_size=(1600, 900))
    app.push(GameScene(MatchSettings(p1_seat="human", p2_seat="bot", seed=seed, first_player=2)))
    g = app.scenes[-1]
    for _ in range(40000):
        g.update(60)
        if g.animator.is_busy or g.bridge.is_over:
            continue
        d = g.bridge.pending_decision
        if d is None or g.panel.decision is not d:
            continue
        game = g.bridge.game
        me = game.players[d.player]
        if d.kind == DecisionKind.CHOOSE_ACTION and len(me.hand) >= 5 and len(me.play_area.creatures) >= 2:
            return app, g
        if d.kind == DecisionKind.CHOOSE_ACTION:
            plays = [o for o in d.options if isinstance(o, PlayCard) and o.card.type.value == "Creature"]
            g.panel._submit(plays[0] if plays else next(o for o in d.options if isinstance(o, EndTurn)))
        elif d.kind in (DecisionKind.CHOOSE_CARDS, DecisionKind.ORDER_EFFECTS):
            g.panel._submit(list(d.options[: max(1, d.min_n)]) if d.kind == DecisionKind.CHOOSE_CARDS else list(d.options))
        else:
            g.panel._submit(False if d.kind == DecisionKind.MULLIGAN else d.options[0])
    return app, g


class TestHitTestingMatchesWhatIsDrawn(unittest.TestCase):
    def test_contains_point_matches_the_rendered_rotation(self):
        app = App(window_size=(1600, 900))
        sprite = CardSprite(1, app.assets)
        sprite.set_size(S.HAND_CARD_W, S.HAND_CARD_H)
        sprite.face_up = False
        for rot in (-12.0, -5.0, 5.0, 12.0):
            sprite.snap_to(400, 400, rot)
            img = pygame.transform.rotozoom(sprite._face_surface(), rot, 1.0)
            mask = pygame.mask.from_surface(img, 127)
            r = img.get_rect(center=(400, 400))
            agree = total = 0
            for x in range(r.left, r.right, 2):
                for y in range(r.top, r.bottom, 2):
                    total += 1
                    agree += bool(mask.get_at((x - r.left, y - r.top))) == sprite.contains_point(x + 0.5, y + 0.5)
            self.assertGreater(agree / total, 0.985, rot)

    def test_card_at_agrees_with_the_topmost_drawn_card(self):
        for seed in (1, 4):
            app, g = _advance_to_busy_board(seed)
            board = g.board
            surf, lookup = _id_map(board)
            agree = total = 0
            for x in range(S.PLAY_X, S.PLAY_X + S.PLAY_W, 3):
                for y in range(0, S.CANVAS_H, 3):
                    c = surf.get_at((x, y))
                    truth = lookup.get((c.r, c.g, c.b))
                    picked = board.card_at((x + 0.5, y + 0.5))
                    if truth is None and picked is None:
                        continue
                    total += 1
                    agree += truth == picked
            self.assertGreater(total, 1000)
            self.assertGreater(agree / total, 0.99, f"seed {seed}: {agree}/{total}")

    def test_clicking_a_glowing_card_picks_exactly_that_card(self):
        app, g = _advance_to_busy_board(1)
        for iid, opts in g.panel.card_option_map.items():
            sprite = g.board.sprites[iid]
            if g.board.card_at((sprite.x, sprite.y)) != iid:
                continue  # its center is covered by another card; card_at already covers overlap
            g.panel.reset()
            g.panel.on_enter(g.bridge.pending_decision, g.bridge.game.view_for(g.viewer), g.board)
            g.panel._dispatch_click((sprite.x, sprite.y), g.board, g.board.layout)
            chosen = g.panel.result if g.panel.result is not None else (g.panel.chooser_opts[0] if g.panel.chooser_opts else g.panel._armed_action)
            self.assertIs(getattr(chosen, "card", None).instance_id, iid)


class TestClickRouting(unittest.TestCase):
    def test_clicking_another_card_while_the_chooser_is_open_acts_on_it(self):  # M4
        for seed in range(1, 16):
            app, g = _advance_to_busy_board(seed)
            p = g.panel
            visible = [iid for iid in p.card_option_map
                       if g.board.card_at((g.board.sprites[iid].x, g.board.sprites[iid].y)) == iid]
            multi = [iid for iid in visible if len(p.card_option_map[iid]) > 1]
            if not multi:
                continue
            a = g.board.sprites[multi[0]]
            p._dispatch_click((a.x, a.y), g.board, g.board.layout)
            self.assertEqual(p.chooser_iid, multi[0])
            chooser = p._chooser_rect(g.board)
            others = [iid for iid in visible if iid != multi[0]
                      and not chooser.collidepoint((g.board.sprites[iid].x, g.board.sprites[iid].y))]
            if others:
                break
        else:
            self.fail("no seed produced a multi-option card next to another reachable legal card")
        b = g.board.sprites[others[0]]
        p._dispatch_click((b.x, b.y), g.board, g.board.layout)
        # The click wasn't swallowed: it either opened B's chooser, picked
        # B's single option, or armed B's confirm-required option.
        acted = (
            p.chooser_iid == others[0]
            or (p.result is not None and getattr(p.result, "card", None).instance_id == others[0])
            or (p._armed_action is not None and p._armed_action.card.instance_id == others[0])
        )
        self.assertTrue(acted)

    def test_upgrade_forwards_clicks_to_its_host(self):  # M5
        app, g = _advance_to_busy_board(1)
        snap = g.board.snapshot
        host = next(cs for cs in snap.cards.values() if cs.zone == "play_creature")
        from gui.snapshot import CardState

        upgrade = CardState(iid=999999, name="Duskrunner", house="Shadows", type="Upgrade", image=None,
                            owner=host.owner, controller=host.owner, zone="upgrade", index=0, zone_count=1,
                            host_iid=host.iid)
        snap.cards[upgrade.iid] = upgrade
        sprite = g.board.sprite_for(upgrade.iid)
        sprite.apply_state(upgrade)
        x, y, rot, w, h = g.board.slot(upgrade, snap)
        sprite.set_size(w, h)
        sprite.snap_to(x, y, rot)
        self.assertEqual(g.board.card_at((x, y)), upgrade.iid)
        self.assertEqual(g.board.click_target_at((x, y)), host.iid)


if __name__ == "__main__":
    unittest.main()
