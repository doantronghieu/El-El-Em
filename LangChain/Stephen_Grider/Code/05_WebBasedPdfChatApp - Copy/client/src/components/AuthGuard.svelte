<script lang="ts">
	import type { User } from '$s/auth';
	import { goto } from '$app/navigation';
	import { onMount } from 'svelte';
	import { auth, getUser } from '$s/auth';

	async function checkAuth(user: User) {
		if (user === false) {
			goto('/auth/signin');
		}
	}
	$: user = $auth.user;
	$: checkAuth($auth.user);

	onMount(() => {
		if (user === null) {
			getUser();
		}
	});
</script>
