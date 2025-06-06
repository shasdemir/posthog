import { IconGear, IconPlus } from '@posthog/icons'
import { useActions, useValues } from 'kea'
import { router } from 'kea-router'
import { upgradeModalLogic } from 'lib/components/UpgradeModal/upgradeModalLogic'
import { FEATURE_FLAGS } from 'lib/constants'
import { LemonButton } from 'lib/lemon-ui/LemonButton'
import { LemonDivider } from 'lib/lemon-ui/LemonDivider'
import { LemonSnack } from 'lib/lemon-ui/LemonSnack/LemonSnack'
import { featureFlagLogic } from 'lib/logic/featureFlagLogic'
import { getProjectSwitchTargetUrl } from 'lib/utils/router-utils'
import { useMemo } from 'react'
import { organizationLogic } from 'scenes/organizationLogic'
import { isAuthenticatedTeam, teamLogic } from 'scenes/teamLogic'
import { urls } from 'scenes/urls'

import { AvailableFeature, TeamBasicType } from '~/types'

import { globalModalsLogic } from '../GlobalModals'
import { EnvironmentSwitcherOverlay } from './EnvironmentSwitcher'

export function ProjectName({ team }: { team: TeamBasicType }): JSX.Element {
    return (
        <div className="flex items-center">
            <span>{team.name}</span>
            {team.is_demo ? <LemonSnack className="ml-2 text-xs shrink-0">Demo</LemonSnack> : null}
        </div>
    )
}

export function ProjectSwitcherOverlay({ onClickInside }: { onClickInside?: () => void }): JSX.Element {
    const { currentOrganization, projectCreationForbiddenReason } = useValues(organizationLogic)
    const { currentTeam } = useValues(teamLogic)
    const { guardAvailableFeature } = useValues(upgradeModalLogic)
    const { showCreateProjectModal } = useActions(globalModalsLogic)
    const { featureFlags } = useValues(featureFlagLogic)

    if (featureFlags[FEATURE_FLAGS.ENVIRONMENTS]) {
        return <EnvironmentSwitcherOverlay onClickInside={onClickInside} />
    }

    return (
        <div className="project-switcher-container">
            <h5>Projects</h5>
            <LemonDivider />
            <CurrentProjectButton onClickInside={onClickInside} />
            {currentOrganization?.teams &&
                currentOrganization.teams
                    .filter((team) => team.id !== currentTeam?.id)
                    .sort((teamA, teamB) => teamA.name.localeCompare(teamB.name))
                    .map((team) => <OtherProjectButton key={team.id} team={team} onClickInside={onClickInside} />)}

            <LemonButton
                icon={<IconPlus />}
                fullWidth
                disabled={!!projectCreationForbiddenReason}
                tooltip={projectCreationForbiddenReason}
                data-attr="new-project-button"
                onClick={() => {
                    onClickInside?.()
                    guardAvailableFeature(AvailableFeature.ORGANIZATIONS_PROJECTS, showCreateProjectModal, {
                        currentUsage: currentOrganization?.teams?.length,
                    })
                }}
            >
                New project
            </LemonButton>
        </div>
    )
}

function CurrentProjectButton({ onClickInside }: { onClickInside?: () => void }): JSX.Element | null {
    const { currentTeam } = useValues(teamLogic)
    const { push } = useActions(router)

    return isAuthenticatedTeam(currentTeam) ? (
        <LemonButton
            active
            sideAction={{
                icon: <IconGear className="text-secondary" />,
                tooltip: `Go to ${currentTeam.name} settings`,
                onClick: () => {
                    onClickInside?.()
                    push(urls.settings('project'))
                },
            }}
            title={`Switch to project ${currentTeam.name}`}
            fullWidth
        >
            <ProjectName team={currentTeam} />
        </LemonButton>
    ) : null
}

function OtherProjectButton({ team }: { team: TeamBasicType; onClickInside?: () => void }): JSX.Element {
    const { location } = useValues(router)
    const { currentTeam } = useValues(teamLogic)

    const relativeOtherProjectPath = useMemo(() => {
        return getProjectSwitchTargetUrl(location.pathname, team.id, currentTeam?.project_id, team.project_id)
    }, [location.pathname, team.id, team.project_id, currentTeam?.project_id])

    return (
        <LemonButton
            to={relativeOtherProjectPath}
            sideAction={{
                icon: <IconGear className="text-secondary" />,
                tooltip: `Go to ${team.name} settings`,
                to: urls.project(team.id, urls.settings()),
            }}
            title={`Switch to project ${team.name}`}
            fullWidth
        >
            <ProjectName team={team} />
        </LemonButton>
    )
}
